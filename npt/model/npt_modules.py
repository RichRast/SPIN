"""Contains base attention modules."""

import math
import pstats
from sparsemax import Sparsemax
import torch
import torch.nn as nn

import pdb


class SaveAttMaps(nn.Module):
    def __init__(self):
        super().__init__()
        self.curr_att_maps = None
        self.Q = None
        self.K = None
        self.V = None
        self.out = None
        self.out_pre_res = None

    def forward(self, X, Q, K, V):
        self.curr_att_maps = nn.Parameter(X)
        self.Q = nn.Parameter(Q)
        self.K = nn.Parameter(K)
        self.V = nn.Parameter(V)

        return X


class MAB(nn.Module):
    """Multi-head Attention Block.

    Based on Set Transformer implementation
    (Lee et al. 2019, https://github.com/juho-lee/set_transformer).
    """
    def __init__(
            self, dim_Q, dim_KV, dim_emb, dim_out, c):
        """

        Inputs have shape (B_A, N_A, F_A), where
        * `B_A` is a batch dimension, along we parallelise computation,
        * `N_A` is the number of samples in each batch, along which we perform
        attention, and
        * `F_A` is dimension of the embedding at input
            * `F_A` is `dim_Q` for query matrix
            * `F_A` is `dim_KV` for key and value matrix.

        Q, K, and V then all get embedded to `dim_emb`.
        `dim_out` is the output dimensionality of the MAB which has shape
        (B_A, N_A, dim_out), this can be different from `dim_KV` due to
        the head_mixing.

        This naming scheme is inherited from set-transformer paper.
        """
        super(MAB, self).__init__()
        mix_heads = c.model_mix_heads
        num_heads = c.model_num_heads
        sep_res_embed = c.model_sep_res_embed
        ln = c.model_att_block_layer_norm
        rff_depth = c.model_rff_depth
        self.att_score_norm = c.model_att_score_norm
        self.pre_layer_norm = c.model_pre_layer_norm
        self.viz_att_maps = c.viz_att_maps
        self.model_ablate_rff = c.model_ablate_rff
        self.c = c
        if self.viz_att_maps:
            self.save_att_maps = SaveAttMaps()

        if dim_out is None:
            dim_out = dim_emb
        elif (dim_out is not None) and (mix_heads is None):
            print('Warning: dim_out transformation does not apply.')
            dim_out = dim_emb

        self.num_heads = num_heads
        self.dim_KV = dim_KV
        self.dim_emb = dim_emb
        
        self.dim_split = dim_emb // num_heads
        if not (self.c.input_shape=='flat' and self.c.att_expansion!=4 and dim_KV!=dim_emb):
            self.fc_q = nn.Linear(dim_Q, dim_emb)
            self.fc_k = nn.Linear(dim_KV, dim_emb)
            self.fc_v = nn.Linear(dim_KV, dim_emb)
        self.fc_mix_heads = nn.Linear(dim_emb, dim_out) if mix_heads else None
        self.fc_res = nn.Linear(dim_Q, dim_out) if sep_res_embed else None

        if ln:
            if self.pre_layer_norm:  # Applied to X
                self.ln0 = nn.LayerNorm(dim_Q, eps=c.model_layer_norm_eps)
            else:  # Applied after MHA and residual
                self.ln0 = nn.LayerNorm(dim_out, eps=c.model_layer_norm_eps)

            self.ln1 = nn.LayerNorm(dim_out, eps=c.model_layer_norm_eps)
        else:
            self.ln0 = None
            self.ln1 = None

        self.hidden_dropout = (
            nn.Dropout(p=c.model_hidden_dropout_prob)
            if c.model_hidden_dropout_prob else None)

        self.att_scores_dropout = (
            nn.Dropout(p=c.model_att_score_dropout_prob)
            if c.model_att_score_dropout_prob else None)
        
        if self.c.SPIN and self.c.input_shape=='flat' and self.c.att_expansion!=4 and (self.dim_KV!=self.dim_emb):
            self.init_short_rff(dim_out)
        else:
            self.init_rff(dim_out, rff_depth)
            self.short_rff = None

        if self.c.sparsemax:
            self.Sparsemax = Sparsemax(dim=2)


    def init_rff(self, dim_out, rff_depth):
        # Linear layer with 4 times expansion factor as in 'Attention is
        # all you need'!
        self.rff = [nn.Linear(dim_out, 4 * dim_out), nn.GELU()]

        if self.hidden_dropout is not None:
            self.rff.append(self.hidden_dropout)

        for i in range(rff_depth - 1):
            self.rff += [nn.Linear(4 * dim_out, 4 * dim_out), nn.GELU()]

            if self.hidden_dropout is not None:
                self.rff.append(self.hidden_dropout)

        self.rff += [nn.Linear(4 * dim_out, dim_out)]

        if self.hidden_dropout is not None:
            self.rff.append(self.hidden_dropout)

        self.rff = nn.Sequential(*self.rff)
    
    def init_short_rff(self, dim_out):
        self.short_rff = [nn.Linear(dim_out, 5), nn.GELU(), nn.Linear(5, dim_out)]
        self.short_rff = nn.Sequential(*self.short_rff)

    def forward(self, X, Y):
        if self.pre_layer_norm and self.ln0 is not None:
            X_multihead = self.ln0(X)
        else:
            X_multihead = X
        
        if not (self.c.SPIN and self.c.input_shape=='flat' and self.c.att_expansion!=4 ):

            Q = self.fc_q(X_multihead)

            if self.fc_res is None:
                X_res = Q
            else:
                X_res = self.fc_res(X)  # Separate embedding for residual

            K = self.fc_k(Y)
            V = self.fc_v(Y)
        else:
            Q=X_multihead
            X_res = Q
            K = Y
            V = Y

        Q_ = torch.cat(Q.split(self.dim_split, 2), 0)
        K_ = torch.cat(K.split(self.dim_split, 2), 0)
        V_ = torch.cat(V.split(self.dim_split, 2), 0)

        # TODO: track issue at
        # https://github.com/juho-lee/set_transformer/issues/8
        # A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        A = torch.einsum('ijl,ikl->ijk', Q_, K_)

        if self.att_score_norm == 'softmax':
            A = torch.softmax(A / math.sqrt(self.dim_KV), 2)
        elif self.att_score_norm == 'constant':
            A = A / self.dim_split
        elif self.c.sparsemax:
            self.Sparsemax(A / math.sqrt(self.dim_KV))
        else:
            raise NotImplementedError
        
        if self.viz_att_maps:
            A = self.save_att_maps(A, Q_, K_, V_)

        # Attention scores dropout is applied to the N x N_v matrix of
        # attention scores.
        # Hence, it drops out entire rows/cols to attend to.
        # This follows Vaswani et al. 2017 (original Transformer paper).

        if self.att_scores_dropout is not None:
            A = self.att_scores_dropout(A)

        multihead = A.bmm(V_)
        multihead = torch.cat(multihead.split(Q.size(0), 0), 2)

        # Add mixing of heads in hidden dim.

        if self.fc_mix_heads is not None:
            H = self.fc_mix_heads(multihead)
        else:
            H = multihead

        # Follow Vaswani et al. 2017 in applying dropout prior to
        # residual and LayerNorm
        if self.hidden_dropout is not None:
            H = self.hidden_dropout(H)

        # True to the paper would be to replace
        # self.fc_mix_heads = nn.Linear(dim_V, dim_Q)
        # and Q_out = X
        # Then, the output dim is equal to input dim, just like it's written
        # in the paper. We should definitely check if that boosts performance.
        # This will require changes to downstream structure (since downstream
        # blocks expect input_dim=dim_V and not dim_Q)

        # Residual connection
        Q_out = X_res
        H = H + Q_out

        # Post Layer-Norm, as in SetTransformer and BERT.
        if not self.pre_layer_norm and self.ln0 is not None:
            H = self.ln0(H)

        if self.pre_layer_norm and self.ln1 is not None:
            H_rff = self.ln1(H)
        else:
            H_rff = H
        
        if self.model_ablate_rff  :
            expanded_linear_H = H_rff
        elif self.short_rff is not None :
            expanded_linear_H = self.short_rff(H_rff)
        else:
            # Apply row-wise feed forward network
            expanded_linear_H = self.rff(H_rff)

        # Residual connection
        expanded_linear_H = H + expanded_linear_H

        if not self.pre_layer_norm and self.ln1 is not None:
            expanded_linear_H = self.ln1(expanded_linear_H)

        if self.viz_att_maps:
            self.save_att_maps.out = nn.Parameter(expanded_linear_H)
            self.save_att_maps.out_pre_res = nn.Parameter(H)

        return expanded_linear_H


class MHSA(nn.Module):
    """
    Multi-head Self-Attention Block.

    Based on implementation from Set Transformer (Lee et al. 2019,
    https://github.com/juho-lee/set_transformer).
    Alterations detailed in MAB method.
    """
    has_inducing_points = False

    def __init__(self, dim_in, dim_emb, dim_out, input_shape, N, c):
        super(MHSA, self).__init__()
        self.c = c
        self.input_shape = input_shape
        c.input_shape = input_shape # to pass this to MAB block 
        self.mab1 = MAB(dim_out, dim_out, dim_emb, dim_out, c)
        if self.c.set_transformer:
            if input_shape=='flat': # ABD, shape of X # (1, N, D.E), D=50, E=64
                self.I = init_latents(c.num_inds_I, dim_in) # (1, I, F.E), I=10, F=10
            elif input_shape=='nested': #ABA, shape of X # (N, D, E)
                self.I = init_latents(c.num_inds_F, dim_in) # (1, F, E)
            self.mab0 = MAB(dim_in, dim_in, dim_emb, dim_out, c)
        elif self.c.perceiver_A:
            pass
        elif self.c.perceiver_C :
            pass
        elif self.c.perceiver_B:
            pass
        elif self.c.perceiver_E:
            self.mab1 = MAB(dim_in, dim_out, dim_emb, dim_in, c)
            if self.c.ABLA :
                self.mab2 = MAB(dim_in, dim_in, dim_emb, dim_in, c)
        elif self.c.perceiver_D :
            self.mab1 = MAB(dim_in, dim_out, dim_emb, dim_in, c)
            if self.c.ABLA :
                self.mab2 = MAB(dim_in, dim_in, dim_emb, dim_in, c)
        elif self.c.SPIN:
            self.mab1 = MAB(dim_in, dim_out, dim_emb, dim_in, c)
            if (self.c.ABLA and (input_shape=='nested')):
                self.mab2 = MAB(dim_in, dim_in, dim_emb, dim_in, c)
        
    def forward(self, X):
        if self.c.set_transformer:
            H=self.mab0(self.I.repeat(X.size(0), 1, 1), X)
            return self.mab1(X, H)
        elif self.c.perceiver_A:
            pass
        elif self.c.perceiver_B:
            pass
        elif self.c.perceiver_C:
            pass
        elif self.c.perceiver_D:
            (X, H) =X
            X, H = self.Perceiver_D( X, H)
            return X, H
        elif self.c.perceiver_E:
            (X, H) =X
            X, H = self.Perceiver_E( X, H)
            return X, H
        elif self.c.SPIN:
            (X, H_ABD, H_ABA) =X
            X, H_ABD, H_ABA = self.SPIN( X, H_ABD, H_ABA)
            return X, H_ABD, H_ABA
        else:
            return self.mab1(X,X) # original full self-attention
    
    def Perceiver_A(self, X, H_ABD, H_ABA):
        pass
    
    def Perceiver_B(self, X, H_ABD, H_ABA):
        pass
    
    def Perceiver_C(self, X, H_ABD, H_ABA):
        pass
    
    def Perceiver_D(self, X, H):
        # one latent H of shape (1, I, F.E)
        if self.input_shape=='flat':
            #ABD
            H = self.mab1(H, X) #(1, I, F.E) with (1, N, D.E)
            # H = self.mab1(H,H)  #(1, I, F.E), optional
        else:    
            #ABLA
            H = H.reshape(H.size(1), self.c.num_inds_F, -1) #(I,F,E)  
            if self.c.ABLA: H = self.mab1(H, H) #(I,F,E)
            #reshape to (1,I, F.E)
            H=H.reshape(1, H.size(0), -1) #(1,I,F.E)
        return X, H
    
    def Perceiver_E(self, X, H_ABA):
        H_ABA = self.mab1(H_ABA, X) # cross attn (N,F,E) with (N,D,E)  
        if self.c.ABLA: H_ABA = self.mab2(H_ABA, H_ABA)
        return X, H_ABA
    
    def SPIN(self, X, H_ABD, H_ABA):
        # one latent H of shape (I, F, E)
        if self.input_shape=='flat':
            #ABD
            #reshape to (1,N, F.E)
            H_ABA=H_ABA.reshape(1, H_ABA.size(0), -1) #(1,N,F.E)
            H_ABD = self.mab1(H_ABD, H_ABA) # cross attn (1, I, F.E) with (1, N, F.E)
            H_ABA=H_ABA.reshape(H_ABA.size(1), self.c.num_inds_F, -1) #(N,F,E)
            
        else:    
            #ABA
            H_ABA = self.mab1(H_ABA, X) # cross attn (N,F,E) with (N,D,E)  
            if self.c.ABLA: H_ABA = self.mab2(H_ABA, H_ABA)
        return X, H_ABD, H_ABA

def init_latents(num_inds, dim_in):
    I = nn.Parameter(torch.Tensor(1, num_inds, dim_in))
    nn.init.xavier_uniform_(I)
    return I


