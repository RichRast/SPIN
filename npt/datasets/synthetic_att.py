import numpy as np
from npt.datasets.base import BaseDataset
import copy
import pdb
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
            
class SyntheticAttDataset(BaseDataset):
    def __init__(self, c):
        super(SyntheticAttDataset, self).__init__(
            fixed_test_set_index=-int(c.dataSynthetic_query/3))   
        self.c = c
        print(f"printing all arguments:{self.c}")
        
    def load(self):
        self.get_data_table()
        self.N, self.D = self.data_table.shape 
        print(f"self.D:{self.D}")
        self.cat_features = list(range(0, self.D))
        self.num_features=[]
        
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)
        
        self.num_target_cols = []
        self.cat_target_cols = list(range(self.D-self.c.dataSynthetic_n_cols, self.D))
        self.is_data_loaded = True
        self.tmp_file_names = ['synthetic_att.csv']
    
    def get_data_table(self):
        n_rows = self.c.dataSynthetic_n_rows #6,000
        n_cols = self.c.dataSynthetic_n_cols #20
        # centroid1 = np.array([0,0,0,1,1,1]*5)
        centroid1 = np.array([1,1,1,1,1,1]*20)
        # centroid2 = np.array([1,1,0,0,1,1]*5)
        centroid2 = np.array([0,0,0,0,0,0]*20)
        # centroid3 = np.array([1,1,1,0,0,0]*5)

        #add bernoulli noise with strength p=0.1
        def addBernoulliNoise(data, p=0.05):
            """
            data is a row vector
            """
            N = data.shape[0]
            mask = np.random.binomial(1, p, N).astype(bool)
            data_new=data.copy()
            data_new[mask] = 1-data[mask]
            
            return data_new

        # n_samples=int(n_rows/3)
        n_samples=int(n_rows/2)
        flip_prob=self.c.bernoulliNoise
        X_1=np.tile(centroid1,(n_samples,1))
        X_1_flip=np.apply_along_axis(addBernoulliNoise, -1, X_1, p=flip_prob)

        Y_1 = np.array([0]*len(X_1))

        X_2=np.tile(centroid2,(n_samples,1))
        X_2_flip=np.apply_along_axis(addBernoulliNoise, -1, X_2)
        Y_2 = np.array([1]*len(X_2))

        # X_3=np.tile(centroid3,(n_samples,1))
        # X_3_flip=np.apply_along_axis(addBernoulliNoise, -1, X_3)
        # Y_3 = np.array([2]*len(X_3))

        # print(X_1_flip.shape, X_2_flip.shape, X_3_flip.shape)
        
        # X=np.concatenate((X_1_flip, X_2_flip, X_3_flip))
        # y_cluster=np.concatenate((Y_1, Y_2, Y_3))

        print(X_1_flip.shape, X_2_flip.shape)
        
        X=np.concatenate((X_1_flip, X_2_flip))
        y_cluster=np.concatenate((Y_1, Y_2))

        #initialize y label as a random binary matrix with 20 dimensions
        y1_label = np.zeros((len(Y_1),n_cols))
        y1_label_new=np.apply_along_axis(addBernoulliNoise, -1, y1_label,p=0.1)
        np.testing.assert_allclose((y1_label_new.sum(axis=-1)/y1_label_new.shape[1]).sum()/len(y1_label_new), 0.1, atol=1e-1)

        y2_label = np.ones((len(Y_2),n_cols))
        y2_label_new=np.apply_along_axis(addBernoulliNoise, -1, y2_label,p=0.1)
        np.testing.assert_allclose((y2_label_new.sum(axis=-1)/y2_label_new.shape[1]).sum()/len(y2_label_new), 1-0.1, atol=1e-1)

        # y3_label = np.zeros((len(Y_3),n_cols))
        # y3_label_new=np.apply_along_axis(addBernoulliNoise, -1, y3_label,p=0.7)
        # np.testing.assert_allclose((y3_label_new.sum(axis=-1)/y3_label_new.shape[1]).sum()/len(y3_label_new), 0.7, atol=1e-1)

        # y_label = np.concatenate((y1_label_new, y2_label_new, y3_label_new))
        y_label = np.concatenate((y1_label_new, y2_label_new))
        print(y_label.shape, y_label, y_label.sum())

        y = np.concatenate((y_cluster.reshape(-1,1), y_label), axis=-1)
        
        #self.c.dataSynthetic_query 2700
        query_rows = int(self.c.dataSynthetic_query/3) 

        X_train_query=np.concatenate((X_1_flip[:100,:].copy() ,X_2_flip[:100,:].copy()))
        X_val_query=np.concatenate((X_1_flip[100:200,:].copy(), X_2_flip[100:200,:].copy()))
        X_test_query=np.concatenate((X_1_flip[200:300,:].copy() ,X_2_flip[200:300,:].copy()))

        y_train_query=np.concatenate((y1_label_new[:100,:].copy() ,y2_label_new[:100,:].copy()))
        y_val_query=np.concatenate((y1_label_new[100:200,:].copy() ,y2_label_new[100:200,:].copy()))
        y_test_query=np.concatenate((y1_label_new[200:300,:].copy() ,y2_label_new[200:300,:].copy()))

        y_cluster_train_query=np.concatenate((Y_1[:100].copy() ,Y_2[:100].copy()))
        y_cluster_val_query=np.concatenate((Y_1[100:200].copy() ,Y_2[100:200].copy()))
        y_cluster_test_query=np.concatenate((Y_1[200:300].copy() ,Y_2[200:300].copy()))

        train_idx = np.arange(len(X_train_query))
        np.random.shuffle(train_idx)
        val_idx = np.arange(len(X_val_query))
        np.random.shuffle(val_idx)
        test_idx = np.arange(len(X_test_query))
        np.random.shuffle(test_idx)
        # print(f'row_index_order test 1st shuffle: {list(test_idx)}')
        
        train_query = np.concatenate((X_train_query, y_train_query), axis=-1)
        val_query = np.concatenate((X_val_query, y_val_query), axis=-1)
        test_query = np.concatenate((X_test_query, y_test_query), axis=-1)
        aux_train = np.concatenate((X, y_label),axis=-1)
        
        train_query=train_query[train_idx]
        val_query=val_query[val_idx]
        test_query=test_query[test_idx]

        self.data_table = np.concatenate((aux_train, train_query, val_query, test_query))
        self.cluster_labels=np.concatenate((y_cluster, y_cluster_train_query[train_idx], y_cluster_val_query[val_idx], y_cluster_test_query[test_idx]))
        self.fixed_split_indices ={}
        self.fixed_split_indices=[np.arange(n_rows+len(train_query)),\
            np.arange(n_rows+len(train_query), n_rows+len(train_query)+len(val_query)),\
            np.arange(n_rows+len(train_query)+len(val_query), self.data_table.shape[0])]