import numpy as np
from npt.datasets.base import BaseDataset
import copy
import pdb
            
class SyntheticDataset(BaseDataset):
    def __init__(self, c):
        super(SyntheticDataset, self).__init__(
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
        self.cat_target_cols = list(range(self.D-20, self.D))
        self.is_data_loaded = True
        self.tmp_file_names = ['synthetic.csv']
    
    def get_data_table(self):
        n_rows = self.c.dataSynthetic_n_rows #5,000
        n_cols = self.c.dataSynthetic_n_cols #50
        binary_matrix=np.zeros((n_rows, n_cols)).astype(bool)
        mask_prob=0.5
        Nm = binary_matrix.shape[0]*binary_matrix.shape[1]
        mask_candidates = np.random.choice(np.arange(Nm), size=int(mask_prob*Nm), replace=False)
        mask_entries = np.nonzero(binary_matrix!=1) # shape Nmx2
        mask_indices_rows = mask_entries[0][mask_candidates]
        mask_indices_cols = mask_entries[1][mask_candidates]
        binary_matrix[mask_indices_rows, mask_indices_cols]=1

        assert binary_matrix.sum() ==  int(mask_prob*Nm)
        query_rows = self.c.dataSynthetic_query #3000
        query_matrix = binary_matrix[:query_rows,:]

        train_query = query_matrix[:int(query_matrix.shape[0]/3),:]
        val_query = query_matrix[int(query_matrix.shape[0]/3):int(2*query_matrix.shape[0]/3),:]
        test_query = query_matrix[int(2*query_matrix.shape[0]/3):,:]
        aux_train = binary_matrix
        self.data_table = np.concatenate((aux_train, train_query, val_query, test_query))
        self.fixed_split_indices ={}
        self.fixed_split_indices=[np.arange(n_rows+int(query_matrix.shape[0]/3))\
        ,np.arange(n_rows+int(query_matrix.shape[0]/3), n_rows+int(2*query_matrix.shape[0]/3) )\
        ,np.arange(n_rows+int(2*query_matrix.shape[0]/3), self.data_table.shape[0])]
        