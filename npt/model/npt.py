"""Contains Tabular Transformer Model definition."""
from itertools import cycle

import torch
import torch.nn as nn

from npt.model.image_patcher import LinearImagePatcher
from npt.model.npt_modules import MHSA, init_latents, MAB
from npt.utils.config_utils import Args
from npt.utils.encode_utils import torch_cast_to_dtype
import pdb

IMAGE_PATCHER_SETTING_TO_CLASS = {
    'linear': LinearImagePatcher,
}


class NPTModel(nn.Module):
    """Non-Parametric Transformers.

    Applies Multi-Head Self-Attention blocks between datapoints,
    and within each datapoint.

    For all model variants, we expect a list of input data, `X_ragged`:
    ```
        len(X_ragged) == N
        X_ragged[i].shape == (D, H_i)
    ```
    In other words, we have `N` input samples. All samples share the same
    number of `D` features, where each feature i is encoded in `H_i`
    dimensions. "Encoding" here refers to the data preprocessing, i.e. the
    one-hot-encoding for categorical features, as well as well as adding
    the mask tokens. (Note that this is done by the code and the user is
    expected to provide datasets as given in `npt.data_loaders`.)

    High-level model overview:

    Initially in NPTModel, `self.in_embedding()` linearly embeds each of the
    `D` feature columns to a shared embedding dimension `E`.
    We learn separate embedding weights for each column.
    This allows us to obtain the embedded data matrix `X_emb` as a
    three-dimensional tensor of shape `(N, D, E)`.
    `E` is referred to as `dim_feat` in the code below.

    After embedding the data, we apply NPT.
    See `build_npt()` for further information.
    NPT applies a series of attention blocks on the input.

    We eventually obtain output of shape `(N, D, E)`,
    which is projected back to the dimensions of the input `X_ragged` using
    `self.out_embedding()`, which applies a learned linear embedding to
    each column `D` separately.
    """
    def __init__(self, c, metadata, device=None):
        """Initialise NPTModel.

        Args:
            c: wandb config
            metadata: Dict, from which we retrieve:
                input_feature_dims: List[int], used to specify the number of
                    one-hot encoded dimensions for each feature in the table
                    (used when reloading models from checkpoints).
                cat_features: List[int], indices of categorical features, used
                    in model initialization if using feature type embeddings.
                num_features: List[int], indices of numerical features, used
                    in model initialization if using feature type embeddings.
                cat_target_cols: List[int], indices of categorical target
                    columns, used if there is a special embedding dimension
                    for target cols.
                num_target_cols: List[int], indices of numerical target
                    columns, used if there is a special embedding dimension
                    for target cols.
            device: Optional[int].
        """
        super().__init__()

        # *** Extract Configs ***
        # cannot deepcopy wandb config.
        if c.mp_distributed:
            self.c = Args(c.__dict__)
        else:
            self.c = Args(c)
        # RR: delete two lines below - for running eval via jupyter nb
        if c.use_notebook:
            self.c = Args(c.__dict__)
            print(self.c)

        # * Main model configuration *
        self.device = device

        # * Dataset Metadata *
        input_feature_dims = metadata['input_feature_dims']
        cat_features = metadata['cat_features']
        num_features = metadata['num_features']
        cat_target_cols = metadata['cat_target_cols']
        num_target_cols = metadata['num_target_cols']
        self.N = metadata["N"]

        # * Dimensionality Configs *
        # how many attention blocks are stacked after each other
        self.stacking_depth = c.model_stacking_depth

        # the shared embedding dimension of each attribute is given by
        self.dim_hidden = c.model_dim_hidden

        # we use num_heads attention heads
        self.num_heads = c.model_num_heads

        # how many feature columns are in the input data
        # apply image patching if specified
        if self.c.model_image_n_patches:
            extra_args = {}

            # num_input_features = n_patches per image
            self.image_patcher = IMAGE_PATCHER_SETTING_TO_CLASS[
                self.c.model_image_patch_type](
                input_feature_dims=input_feature_dims,
                dim_hidden=self.dim_hidden,
                c=c, **extra_args)
            npt_attrs = self.image_patcher.get_npt_attrs()
            for k, v in npt_attrs.items():
                self.__setattr__(name=k, value=v)
        else:
            self.image_patcher = None
            self.num_input_features = len(input_feature_dims)
        # whether or not to add a feature type embedding
        self.use_feature_type_embedding = c.model_feature_type_embedding

        # whether or not to add a feature index embedding
        self.use_feature_index_embedding = c.model_feature_index_embedding

        # *** Build Model ***

        # We immediately embed each element
        # (i.e., a table with N rows and D columns has N x D elements)
        # to the hidden_dim. Similarly, in the output, we will "de-embed"
        # from this hidden_dim.

        # Build encoder
        
        self.enc = self.get_npt()
        # *** Dropout and LayerNorm in In-/Out-Embedding ***

        # Hidden dropout is applied for in- and out-embedding
        self.embedding_dropout = (
            nn.Dropout(p=c.model_hidden_dropout_prob)
            if c.model_hidden_dropout_prob else None)

        # LayerNorm applied after embedding, before dropout
        if self.c.embedding_layer_norm and device is None:
            print(
                'Must provide a device in NPT initialization with embedding '
                'LayerNorm.')
        elif self.c.embedding_layer_norm:
            # we batch over rows and columns
            # (i.e. just normalize over E)
            layer_norm_dims = [self.dim_hidden]
            self.embedding_layer_norm = nn.LayerNorm(
                layer_norm_dims, eps=self.c.model_layer_norm_eps)
        else:
            self.embedding_layer_norm = None

        # *** Input In/Out Embeddings ***
        # Don't use for Image Patching - those are handled by the respective
        # init_image_patching

        # In-Embedding
        # Linearly embeds each of the `D` [len(input_feature_dims)] feature
        # columns to a shared embedding dimension E [dim_hidden].
        # Before the embedding, each column has its own dimensionionality
        # H_j [dim_feature_encoding], given by the encoding dimension of the
        # feature (e.g. This is given by the one-hot-encoding size for
        # categorical variables + one dimension for the mask token and two-
        # dimensional for continuous variables (scalar + mask_token)).
        # See docstring of NPTModel for further context.

        if self.image_patcher is None:
            self.in_embedding = nn.ModuleList([
                nn.Linear(dim_feature_encoding, self.dim_hidden)
                for dim_feature_encoding in input_feature_dims])

        # Feature Type Embedding
        # Optionally, we construct "feature type" embeddings -- i.e. we learn a
        # representation based on whether the feature is either
        # (i) numerical or (ii) categorical.
        if self.use_feature_type_embedding:
            if cat_features is None or num_features is None:
                raise Exception(
                    'Must provide cat_feature and num_feature indices at '
                    'NPT initialization if you aim to compute feature type'
                    ' embeddings.')

            if c.mp_distributed and device is None:
                raise Exception(
                    'Must provide device to NPT initialization: in '
                    'distributed setting, and aim to do feature type '
                    'embedding.')

            # If all features are either categorical or numerical,
            # don't bother.
            if len(cat_features) == 0 or len(num_features) == 0:
                print(
                    'All features are either categorical or numerical. '
                    'Not going to bother doing feature type embeddings.')
                self.feature_type_embedding = None
            else:
                self.feature_types = torch_cast_to_dtype(torch.empty(
                    self.num_input_features, device=device), 'long')

                for feature_index in range(self.num_input_features):
                    if feature_index in num_features:
                        self.feature_types[feature_index] = 0
                    elif feature_index in cat_features:
                        self.feature_types[feature_index] = 1
                    else:
                        raise Exception

                self.feature_type_embedding = nn.Embedding(
                    2, self.dim_hidden)

            print(
                f'Using feature type embedding (unique embedding for '
                f'categorical and numerical features).')
        else:
            self.feature_type_embedding = None

        # Feature Index Embedding
        # Optionally, learn a representation based on the index of the column.
        # Allows us to explicitly encode column identity, as opposed to
        # producing this indirectly through the per-column feature embeddings.
        if self.use_feature_index_embedding:
            if c.mp_distributed and device is None:
                raise Exception(
                    'Must provide device to NPT initialization: in '
                    'distributed setting, and aim to do feature index '
                    'embedding.')

            self.feature_indices = torch_cast_to_dtype(
                torch.arange(self.num_input_features, device=device), 'long')

            self.feature_index_embedding = nn.Embedding(
                self.num_input_features, self.dim_hidden)

            print(
                f'Using feature index embedding (unique embedding for '
                f'each column).')
        else:
            self.feature_index_embedding = None

        # Out embedding.
        # The outputs of the AttentionBlocks have shape (N, D, E)
        # [N, len(input_feature_dim), dim_hidden].
        # For each of the column j, we then project back to the dimensionality
        # of that column in the input (N, H_j-1), subtracting 1, because we do
        # not predict the mask tokens, which were present in the input.

        if self.image_patcher is None:
            # Need to remove the mask column if we are using BERT augmentation,
            # otherwise we just project to the same size as the input.
            if self.c.model_bert_augmentation:
                get_dim_feature_out = lambda x: x - 1
            else:
                get_dim_feature_out = lambda x: x

            self.out_embedding = nn.ModuleList([
                nn.Linear(
                    self.dim_hidden,
                    get_dim_feature_out(dim_feature_encoding))
                for dim_feature_encoding in input_feature_dims])

        # *** Gradient Clipping ***
        if c.exp_gradient_clipping:
            clip_value = c.exp_gradient_clipping
            print(f'Clipping gradients to value {clip_value}.')
            for p in self.parameters():
                p.register_hook(
                    lambda grad: torch.clamp(grad, -clip_value, clip_value))

    def get_npt(self):
        """
        A model performing "flattened" attention over the rows and
        "nested" attention over the columns.

        This is reasonable if we don't aim to maintain column equivariance
        (which we essentially never do, because of the column-specific
        feature embeddings at the input and output of the NPT encoder).

        This is done by concatenating the feature outputs of column
        attention and inputting them to row attention. Therefore, it requires
        reshaping between each block, splitting, and concatenation.
        """
        if self.c.allow_even_only_stacking_depth:
            if self.stacking_depth < 2:
                raise ValueError(
                    f'Stacking depth {self.stacking_depth} invalid.'
                    f'Minimum stacking depth is 2.')
            if self.stacking_depth % 2 != 0:
                raise ValueError('Please provide an even stacking depth.')

        print('Building NPT.')

        # *** Construct arguments for row and column attention. ***

        row_att_args = {'c': self.c}
        col_att_args = {'c': self.c}

        # Perform attention over rows first
        att_args = cycle([row_att_args, col_att_args])
        AttentionBlocks = cycle([MHSA])

        D = self.num_input_features
        self.x_datapoints=self.N if self.c.exp_batch_size==-1 else self.c.exp_batch_size
        self.D = D
        enc = []

        if self.c.model_hybrid_debug:
            enc.append(Print())

        # Reshape to flattened representation (1, N, D*dim_input) if starting with 'ABD'
        if (self.c.startAttBlock=='ABD' or self.c.onlyABD) and not (self.c.onlyABA or self.c.perceiver_E): 
            enc.append(ReshapeToFlat())
        enc = self.build_hybrid_enc(
            enc, AttentionBlocks, att_args, D)

        enc = nn.Sequential(*enc)
        
        if self.c.perceiver_A or self.c.perceiver_B or self.c.perceiver_C :
            pass
        elif self.c.perceiver_D :
            if self.c.num_inds_I< self.x_datapoints or self.c.num_inds_F < self.D:
                self.convert_X = MAB(D*self.dim_hidden, int(self.c.num_inds_F*self.dim_hidden*self.c.factor_embed_ABD), D*self.dim_hidden, D*self.dim_hidden, self.c)
        elif self.c.perceiver_E:
            if self.c.num_inds_F < self.D:
                self.convert_X_to_ABA = nn.Linear(self.D, self.c.num_inds_F)
                self.convert_X = MAB(self.dim_hidden, self.dim_hidden, self.dim_hidden, self.dim_hidden, self.c)
        elif self.c.SPIN:
            if self.c.num_inds_I< self.x_datapoints or self.c.num_inds_F < self.D:
                self.convert_X_to_ABA = nn.Linear(self.D, self.c.num_inds_F)
                if self.c.decoderBoth :
                    self.convert_X_ABA = MAB(self.dim_hidden, self.dim_hidden, self.dim_hidden, self.dim_hidden, self.c)
                    self.convert_X_ABD = MAB(D*self.dim_hidden, int(self.c.num_inds_F*self.dim_hidden*self.c.factor_embed_ABD), D*self.dim_hidden, D*self.dim_hidden, self.c)
                elif self.c.startAttBlock=='ABD' and (not self.c.onlyABD or self.c.onlyABA):
                    # if starting with ABD, will end with ABA
                    # self.convert_X_ABA = MAB(self.dim_hidden, self.dim_hidden, self.dim_hidden, self.dim_hidden, self.c)
                    raise ValueError('Please start with ABA for this model')
                else:
                    self.convert_X_ABD = MAB(D*self.dim_hidden, int(self.c.num_inds_F*self.dim_hidden*self.c.factor_embed_ABD), D*self.dim_hidden, D*self.dim_hidden, self.c)
                    
        
        return enc

    def build_hybrid_enc(self, enc, AttentionBlocks, att_args, D):
        if self.c.model_hybrid_debug:
            stack = [Print()]
        else:
            stack = []

        self.initAttBlock()
        if (self.c.perceiver_A or self.c.perceiver_B  or self.c.perceiver_C or self.c.perceiver_D) and (self.c.num_inds_I < self.x_datapoints or self.c.num_inds_F < self.D):
            self.dim_in_flat_Q=int(self.dim_hidden * self.c.num_inds_F * self.c.factor_embed_ABD)
            self.dim_in_flat_KV=self.dim_hidden * D
            self.dim_embed_flat = int(self.dim_hidden * self.c.num_inds_F * self.c.factor_embed_ABD)
            self.dim_in_nested=self.dim_hidden 
        
        elif self.c.perceiver_E and self.c.num_inds_F < self.D:
            self.dim_in_flat_Q=self.dim_hidden 
            self.dim_in_flat_KV=self.dim_hidden
            self.dim_embed_flat = self.dim_hidden
            self.dim_in_nested=self.dim_hidden 

        elif self.c.SPIN and (self.c.num_inds_F < self.D):
            self.dim_in_flat_Q=int(self.dim_hidden * self.c.num_inds_F * self.c.factor_embed_ABD)
            self.dim_in_flat_KV=self.dim_hidden * self.c.num_inds_F
            self.dim_embed_flat = int(self.dim_hidden * self.c.num_inds_F * self.c.factor_embed_ABD)
            self.dim_in_nested=self.dim_hidden 

        elif self.c.SPIN and (self.c.num_inds_F >= self.D):
            self.dim_in_flat_Q=self.dim_hidden * self.c.num_inds_F
            self.dim_embed_flat = self.dim_hidden * self.D
            self.dim_in_flat_KV=self.dim_hidden * self.D
            self.dim_in_nested=self.dim_hidden 
            
        elif self.c.onlyABD:
            self.dim_in_flat_Q=self.dim_hidden * D
            self.dim_in_flat_KV=self.dim_hidden * D
            self.dim_embed_flat = self.dim_hidden * D
            self.dim_in_nested=self.dim_hidden * D

        elif self.c.onlyABA:
            self.dim_in_flat_Q=self.dim_hidden 
            self.dim_in_flat_KV=self.dim_hidden 
            self.dim_embed_flat = self.dim_hidden
            self.dim_in_nested=self.dim_hidden 
        else:
            self.dim_in_flat_Q=self.dim_hidden * D
            self.dim_in_flat_KV=self.dim_hidden * D
            self.dim_embed_flat = self.dim_hidden * D
            self.dim_in_nested=self.dim_hidden

        while self.layer_index < self.stacking_depth:
            
            if self.layer_index % 2 == 1:
                # Input is already in nested shape (N, D, E)
                stack.append(next(AttentionBlocks)(
                    self.dim_in_nested, self.dim_in_nested, self.dim_in_nested, self.final_shape, self.N,
                    **next(att_args)))

                # Reshape to flattened representation
            
                if not (self.c.onlyABD or self.c.onlyABA or self.c.perceiver_E): stack.append(ReshapeToFlat())
                self.final_shape = 'flat'

                if self.c.model_hybrid_debug:
                    stack.append(Print())
            else:
                # Input is already in flattened shape (1, N, D*E)

                # Attend between instances N
                # whenever we attend over the instances,
                # we consider dim_hidden = self.c.dim_hidden * D
                stack.append(next(AttentionBlocks)(
                    self.dim_in_flat_Q, self.dim_embed_flat,
                    self.dim_in_flat_KV, self.final_shape, self.N,
                    **next(att_args)))

                # Reshape to nested representation
              
                if not (self.c.onlyABD or self.c.onlyABA or self.c.perceiver_E): stack.append(ReshapeToNested(D=D))
                self.final_shape = 'nested'

                if self.c.model_hybrid_debug:
                    stack.append(Print())

            # Conglomerate the stack into the encoder thus far
            enc += stack
            stack = []

            self.layer_index += 1

        # Reshape to nested representation, for correct treatment
        # after enc
        if self.final_shape == 'flat' and not (self.c.onlyABD or self.c.onlyABA or self.c.perceiver_E):
            enc.append(ReshapeToNested(D=D))
    
        return enc

    def forward(self, X_ragged, X_labels=None, eval_model=None, dataset_mode=None):
        if self.image_patcher is not None:
            X = self.image_patcher.encode(X_ragged)
            in_dims = [X.size(0), X.size(1), -1]
        else:
            in_dims = [X_ragged[0].shape[0], len(X_ragged), -1]

            # encode ragged input array D x {(NxH_j)}_j to NxDxE)
            X = [embed(X_ragged[i]) for i, embed in enumerate(self.in_embedding)]
            X = torch.stack(X, 1)
        
        # Compute feature type (cat vs numerical) embeddings, and add them
        if self.feature_type_embedding is not None:
            feature_type_embeddings = self.feature_type_embedding(
                self.feature_types)

            # Add a batch dimension (the rows)
            feature_type_embeddings = torch.unsqueeze(
                feature_type_embeddings, 0)

            # Tile over the rows
            feature_type_embeddings = feature_type_embeddings.repeat(
                X.size(0), 1, 1)

            # Add to X
            X = X + feature_type_embeddings

        # Compute feature index embeddings, and add them
        if self.feature_index_embedding is not None:
            feature_index_embeddings = self.feature_index_embedding(
                self.feature_indices)

            # Add a batch dimension (the rows)
            feature_index_embeddings = torch.unsqueeze(
                feature_index_embeddings, 0)

            # Tile over the rows
            feature_index_embeddings = feature_index_embeddings.repeat(
                X.size(0), 1, 1)

            # Add to X
            X = X + feature_index_embeddings

        # Embedding tensor currently has shape (N x D x E)

        # Follow BERT in applying LayerNorm -> Dropout on embeddings
        if self.embedding_layer_norm is not None:
            X = self.embedding_layer_norm(X)

        if self.embedding_dropout is not None:
            X = self.embedding_dropout(X)

        # apply NPT
        
        if self.c.perceiver_A or self.c.perceiver_B or self.c.perceiver_C:
            pass

        elif self.c.perceiver_D:
            if (self.c.num_inds_I<X.shape[0] or self.c.num_inds_F < self.D) :
                
                H_ABD = self.ABD.repeat(1,1,1).to(self.device)
                if dataset_mode =='train' or self.c.enableEncoder: 
                    X, H_ABD = self.enc((X, H_ABD))
                #decoder
                # H is of shape (1, I, F.E) , reshape X to flat form (1, N, D.E) regardless of ABD or ABA
                X = X.reshape(1, X.size(0), -1)
                X = self.convert_X(X, H_ABD) # cross-attn of (1, N, D.E) with (1, I, F.E)
                #reshape X back to (N, D, E)
                X = X.reshape(X.size(1), self.D, -1)
            else: X = self.enc(X)
        
        elif self.c.perceiver_E:
            if self.c.num_inds_F < self.D :
                
                # H_ABA = self.ABA.repeat(X.size(0),1,1).to(self.device)
                H_ABA = self.convert_X_to_ABA(X.permute(0,2,1).contiguous()).permute(0,2,1).contiguous() # linear from N,D,E to N,F,E
                if dataset_mode =='train' or self.c.enableEncoder: 
                    X, H_ABA = self.enc((X, H_ABA))
                #decoder
                # H is of shape (N, F, E)
                X = self.convert_X(X, H_ABA) # cross-attn of (N, D,E) with (N, F, E)
            else: X = self.enc(X)

        elif self.c.SPIN:
            if self.c.num_inds_F < self.D :
                H_ABA = self.convert_X_to_ABA(X.permute(0,2,1).contiguous()).permute(0,2,1).contiguous() # linear from N,D,E to N,F,E
            else: 
                H_ABA = X
            if self.c.num_inds_I<X.shape[0] or self.c.num_inds_F < self.D:
                H_ABD = self.ABD.repeat(1,1,1).to(self.device)#(1,I, F.E)

            else:
                H_ABD = X
            if dataset_mode =='train' or self.c.enableEncoder:
                X, H_ABD, H_ABA = self.enc((X, H_ABD, H_ABA))
            
            #decoder
            if self.final_shape=='flat':
                X=self.convert_X_ABA(X, H_ABA)
            else: # end at ABD, nested final shape
                if self.c.decoderBoth :
                    X=self.convert_X_ABA(X, H_ABA)
                X=X.reshape(1, X.size(0), -1) # reshape from nested to flat
                X=self.convert_X_ABD(X, H_ABD)
                X = X.reshape(X.size(1), self.D, -1) # reshape back to nested
        else:
            X = self.enc(X)

            if X.shape[1] == in_dims[0]:
                # for uneven stacking_depth, need to permute one last time
                # to obtain output of shape (N, D, E)
                X = X.permute([1, 0, 2])

        # Dropout before final projection (follows BERT, which performs
        # dropout before e.g. projecting to logits for sentence classification)
        if self.embedding_dropout is not None:
            X = self.embedding_dropout(X)

        if self.image_patcher is None:
            # project back to ragged (dimensions D x {(NxH_j)}_j )
            # Is already split up across D
            X_ragged = [de_embed(X[:, i]) for i, de_embed in enumerate(
                self.out_embedding)]
        else:
            X_ragged = self.image_patcher.decode(X)

        return X_ragged

    def initAttBlock(self):
        if self.c.startAttBlock=='ABD' and not self.c.onlyABA:
            self.final_shape='flat'
            self.layer_index=0
        else:
            self.final_shape='nested'
            self.layer_index=1
            self.stacking_depth +=1
        if self.c.perceiver_D or self.c.SPIN:
            self.ABD = init_latents(self.c.num_inds_I, int(self.dim_hidden*self.c.num_inds_F*self.c.factor_embed_ABD))


class Permute(nn.Module):
    """Permutation as nn.Module to include in nn.Sequential."""
    def __init__(self, idxs):
        super(Permute, self).__init__()
        self.idxs = idxs

    def forward(self, X):
        return X.permute(self.idxs)


class ReshapeToFlat(nn.Module):
    """Reshapes a tensor of shape (N, D, E) to (1, N, D*E)."""
    def __init__(self):
        super(ReshapeToFlat, self).__init__()

    @staticmethod
    def forward(X):
        if isinstance(X, tuple):
            if len(X)==3:
                # separate latents
                X, H_ABD, H_ABA=X
                return X.reshape(1, X.size(0), -1), H_ABD, H_ABA
            else:
                #one latent
                X, H=X
                return X.reshape(1, X.size(0), -1), H
        return X.reshape(1, X.size(0), -1)


class ReshapeToNested(nn.Module):
    """Reshapes a tensor of shape (1, N, D*E) to (N, D, E)."""
    def __init__(self, D):
        super(ReshapeToNested, self).__init__()
        self.D = D

    def forward(self, X):
        if isinstance(X, tuple):
            if len(X)==3:
                # separate latents
                X, H_ABD, H_ABA=X
                return X.reshape(X.size(1), self.D, -1), H_ABD, H_ABA
            else:
                #one latent
                X, H=X
                return X.reshape(X.size(1), self.D, -1), H
        return X.reshape(X.size(1), self.D, -1)


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print('Debug', x.shape)
        return x
