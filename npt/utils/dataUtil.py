import numpy as np
from scipy.stats import pearsonr
import torch
import itertools
import math
import pdb
import os
import pandas as pd
import pickle
from npt.utils.decorators import timer

def makeDict(win_size):
    dict_keys = list(map(list, itertools.product([0, 1], repeat=win_size)))
    dict_keys = [''.join([str(i) for i in k]) for k in dict_keys]
    dict_values = np.arange(math.pow(2,win_size)).astype(int)
    vocab={k:v+1 for k,v in zip(dict_keys, dict_values)} # ignore 0,1 as 0,1 is used for masking labels
    return vocab

def mapKmer(batchSeq, win_size, win_stride):
    batchSeqKmer = formKmer(batchSeq, win_size, win_stride).astype(int)
    batchSeqKmer=np.apply_along_axis(lambda x: ''.join([str(i) for i in x]), 2, batchSeqKmer)
    posSeq = np.vectorize(makeDict(win_size).get)(batchSeqKmer)
    return posSeq

def formKmer(batchSeq, win_size, win_stride):
    return np.lib.stride_tricks.sliding_window_view(batchSeq,win_size,1)[:,::win_stride]
    
def getPerSnpProbs(ProbWords, mask , win_size):
    """
    When predicting KMer words, return the snp probabilities by marginalizing across all posssible
    combination of words
    Args:
        ProbWords: probability of each word obtained after softmax
        mask: mask matrix representing which position is 1 for a particular word
        win_size: window-size or length of the Kmer word
    Returns:
        probs: per snp probability 
    """
    ProbWords = ProbWords.unsqueeze(dim=2)
    ProbWords=ProbWords.repeat(1,1,win_size,1)
    mask = mask.unsqueeze(2).unsqueeze(0)
    ProbWords_new = ProbWords*mask
    prob = torch.sum(ProbWords_new, dim=1)
    #assert that snp prob sum to 1
    assert torch.isclose(prob + torch.sum(ProbWords*(1-mask)), torch.ones_like(prob)).sum(), "snp prob does not sum to 1"
    prob=prob.permute(0,2,1).contiguous()
    prob=prob.reshape(ProbWords.shape[0],-1)

    return prob

def getTensorDifference(tensor1, tensor2):
    """
    Return tensor 1 that is not common with tensor 2
    """
    combined_tensors = torch.cat((tensor1, tensor2))
    uniques, counts = combined_tensors.unique(return_counts=True, dim=0)
    difference=uniques[counts==1]
    intersection=uniques[counts>1]
    return difference

@timer
def getPerSnpPreds_Labels(probs, label):
    """
    Args:
        probs shape: [n_samples*2, n_unknown_variants]
        labels shape: [n_samples*2, n_unknown_variants]
    Returns:
        hap1 pred [n_samples, n_unknown_variants], 
        hap2 pred, hap1 label, hap2 label, 
        pearsonr2_per_snp [n_unknown_variants], 
        overall_pearsonr2 (scaler)
        
    """
    n_unknown_variants=probs.shape[1]
    probs = probs.reshape(-1,2,n_unknown_variants) # n_samples, diploid, n_variants
    probs = np.transpose(probs,(0,2,1)) # n_samples, n_variants, diploid
    label = label.reshape(-1,2,n_unknown_variants)
    label = np.transpose(label,(0,2,1))
    hap1_pred = probs[:,:,0] # n_samples, n_variants
    hap2_pred = probs[:,:,1] # n_samples, n_variants
    hap1_label = label[:,:,0]
    hap2_label = label[:,:,1]
    pearsonrFunc = np.vectorize(pearsonr,signature='(n),(n)->(),()')
    preds = np.concatenate((hap1_pred, hap2_pred),axis=0) # n_samples*2, n_variants
    labels = np.concatenate((hap1_label, hap2_label),axis=0) # n_samples*2, n_variants
    pearsonr2_per_snp = pearsonrFunc(preds.T, labels.T)[0]**2 # n_variants
    overall_pearsonr2 = pearsonr(preds.flatten(), labels.flatten())[0]**2 # scalar value
    return hap1_pred, hap1_label, hap2_pred, hap2_label, pearsonr2_per_snp, overall_pearsonr2

@timer
def getResultsDict(unknown_snp_ids, hap1_test_pred, hap2_test_pred, hap1_test_label, hap2_test_label, test_pearsonr2_per_snp,**kwargs):
    hap1_train_label=kwargs.get('hap1_train_label') # n_samples, n_variants
    hap2_train_label=kwargs.get('hap2_train_label')
    train_per_snp_r2=kwargs.get('train_per_snp_r2')    
    mac = hap1_train_label.sum(axis=0) + hap1_test_label.sum(axis=0) + hap2_train_label.sum(axis=0) + hap2_test_label.sum(axis=0) 
    maf = mac/(hap1_train_label.shape[0] + hap1_test_label.shape[0])
    resultsDict={}
    for i, snpId in enumerate(unknown_snp_ids):
        resultsDict[snpId]={
        "Minor_allele_count":  mac[i],
        "Minor_allele_frequency": maf[i],
        "test pearson r^2":test_pearsonr2_per_snp[i],
        "train pearson r^2":train_per_snp_r2[i],
        "hap1 preds": hap1_test_pred[:,i],
        "hap1 labels": hap1_test_label[:,i],
        "hap2 preds":hap2_test_pred[:,i],
        "hap2 labels":hap2_test_label[:,i]

        }
    return resultsDict

def deleteIdxTensor(Tensor, indices, dimToRem):
    """
    Tensor is 3d
    indices: 1d Tensor
    dimToRem: shape to remove starting with 0
    """
    all_indices=np.arange(Tensor.shape[dimToRem])
    indices = indices.cpu().detach().numpy()
    selectIdx=np.setdiff1d(all_indices, indices)
    return torch.index_select(Tensor, dimToRem, torch.tensor(selectIdx).type(torch.int32))

def getCudaFreeMem():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    # in Gb
    divisorGb=1024 ** 3
    r=r/divisorGb
    a=a/divisorGb
    f=f/divisorGb

    return [f, r, a]

def getCudaPeakMem():
    stats = torch.cuda.memory_stats()
    peak_bytes_requirement = stats["allocated_bytes.all.peak"]
    return peak_bytes_requirement / 1024 ** 3
    # print(f"Peak memory requirement: {peak_bytes_requirement / 1024 ** 3:.2f} GB")

class Running_Average():
    def __init__(self):
        self.reset()
    
    def update(self, val, step_size):
        self.value += val
        self.steps += step_size
    
    def reset(self):
        self.value=0
        self.steps=0
        
    def __call__(self):
        return self.value/float(self.steps)