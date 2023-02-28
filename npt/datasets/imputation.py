import numpy as np
import torch
import pandas as pd 
from torch.utils.data import Dataset
import os
import pickle
import itertools
import pdb
from npt.utils.dataUtil import mapKmer
from npt.datasets import vcf_processor
from npt.datasets.vcf_processor import VCF_Processing, split_samples, read_predefined_variants
import os.path as osp
from npt.utils.decorators import timer
from npt.datasets.base import BaseDataset

DEFAULT_WINDOW_SUFFIX = {
    0: [1,1,1,1,1,1],
    1: [0,0,0,0,0,0],
    2: [1,0,1,0,1,0],
    3: [0,1,0,1,0,1],
    4: [1,1,0,0,1,1],
    5: [0,0,1,1,0,0],
    6: [1,1,1,0,0,0],
    7: [0,0,0,1,1,1],
    8: [1,0,1,1,1,0],
    9: [1,1,0,0,0,1],
    10: [0,0,1,1,1,0],
}
# experiments ran with  5 random chm 20 windows 5430, 2876, 1695, 4246,  683
DEFAULT_WIN = {
    0: '06',
    1: '16',
    2: '28',
    3: '42',
    4: '54'
    }
DEFAULT_SUBWIN = {
        0: 83,
        1: 95,
        2: 76,
        3: 46,
        4: 30
        }
def formWinDicts(num_win, win_size):
    """
    First 5 are defined as above by default. this code is 
    when num_win>5 and extends the default_win and default_subwin dict
    """
    if win_size==100:
        existing_win=[5430, 2876, 1695, 4246, 683]
        win_arr = np.random.choice(np.setdiff1d(np.arange(6600), existing_win), size=num_win-5, replace=False)
        for i, val in enumerate(win_arr):
            tmp_strList = list(str(val))
            if 4-len(tmp_strList)>0:
                tmp_strList=['0']*(4-len(tmp_strList))+tmp_strList
            DEFAULT_WIN[i+5]=tmp_strList[0]+tmp_strList[1]
            DEFAULT_SUBWIN[i+5]=int(tmp_strList[2]+tmp_strList[3])
    else:
        DEFAULT_WIN.clear()
        DEFAULT_SUBWIN.clear()
        win_super = np.random.choice(np.arange(66), size=num_win, replace=False)
        win_sub = np.random.choice(np.arange(int(10000/win_size)-2), size=num_win, replace=False) # -2 is a hack, because some subsets don't contain full 10K
        for i, (num_super, num_sub) in enumerate(zip(win_super, win_sub)):
            tmp_strSuper = list(str(num_super))
            if len(tmp_strSuper)<2:
                tmp_strSuper=['0']*(2-len(tmp_strSuper))+tmp_strSuper
            DEFAULT_WIN[i]=tmp_strSuper[0]+tmp_strSuper[1]
            DEFAULT_SUBWIN[i]=num_sub
    print(f"DEFAULT_WIN:{DEFAULT_WIN}")
    print(f"DEFAULT_SUBWIN:{DEFAULT_SUBWIN}")

class Haplotype():
    @timer
    def __init__(self, n_closest, data_dir, vcf_filename, marker_path, unknown_snp_list, chromosome='20', **kwargs):
        self.data_dir = data_dir
        self.vcf_filename = vcf_filename
        self.chm=chromosome
        self.marker_path=marker_path
        self.unknown_snp_list=unknown_snp_list
        # other kwargs option
        self.npt = kwargs.get('npt')
        self.subWin_size = kwargs.get('subWin_size')
        self.dataset_iter = kwargs.get('dataset_iter')
        self.old = kwargs.get('old')
        # get the data object
        self.loadVcf()
        #split samples into train, val, test
        self.train_samples, self.val_samples, self.test_samples = self.getSampleSplit()
        self.known_variants, self.unknown_variants = self.getKnownUnknownVariants()
        self.unknown_snp_ids = [self.data.get_all_variant_ids()[idx] for idx in self.unknown_variants]
        print(f"snp ids to impute:{self.unknown_snp_ids}")
        # get closest snps
        self.getClosestSnps(n_closest=n_closest)

    @timer
    def loadVcf(self):
        # initialize self.data here
        vcf_file_path = os.path.join(self.data_dir, self.vcf_filename.format(self.chm))
        print("Parsing vcf file {}".format(os.path.join(self.data_dir, vcf_file_path))) 
        tmp_file = os.path.join(os.environ.get('DATA_IN'),'chr{}.pkl'.format(self.chm))
        if os.path.exists(tmp_file):
            with open(tmp_file, 'rb') as f:
                import sys
                sys.modules['vcf_processor'] = vcf_processor
                self.data = pickle.load(f)
            print("loaded cached version of data from {}".format(tmp_file))
        else:
            self.data = VCF_Processing(vcf_file=vcf_file_path, chromosome=self.chm, alt_number=6)
            with open(tmp_file, 'wb') as f:
                pickle.dump(self.data, f)
            print("cached version of data to {}".format(tmp_file))
        print("loaded serialized file")
        #------------------- Filter variants by minor allele count -------------#
        filtered_variants = self.data.select_variants(1)#minor_allele_count >= 1
        self.data.filter_variants(filtered_variants)
        print("Filtered variants")
        self.data.build_indicies()

    def getSampleSplit(self):
        #------------------- Split train test as defined in file -------------#
        test_samples = pd.read_csv(os.path.join(os.environ.get('DATA_IN'),'test_samples.csv'), index_col=0, squeeze=True)
        sample_series = pd.Series(self.data.get_samples())
        test_membership, ref_membership = split_samples(sample_series, test_samples)
        test_samples = sample_series[test_membership]
        train_valid_samples = sample_series[ref_membership]

        # if val samples are present
        val_samples = None
        val_samples_path = os.path.join(os.environ.get('DATA_IN'),'val_samples.csv')
        if os.path.exists(val_samples_path):
            val_samples = pd.read_csv(val_samples_path, index_col=0, squeeze=True)
            val_membership, train_membership = split_samples(train_valid_samples, val_samples)
            val_samples = train_valid_samples[val_membership]
            train_samples = train_valid_samples[train_membership]
        print(f" number of train samples:{len(train_samples)}, val samples:{len(val_samples)}, test_samples:{len(test_samples)}")
        train_samples.to_csv(os.path.join(os.environ.get('DATA_IN'),'train_samples_for_script_training.csv'))
        val_samples.to_csv(os.path.join(os.environ.get('DATA_IN'),'val_samples_for_script_training.csv'))
        test_samples.to_csv(os.path.join(os.environ.get('DATA_IN'),'test_samples_for_script_training.csv'))
        return train_samples, val_samples, test_samples
        
    @timer
    def getKnownUnknownVariants(self):
        #----------------- Read known and unknown from predefined microarray panel ------#

        # get the filename of snp list without csv
        # The unknown variant files are 10K each in size
        snp_filename = osp.splitext(osp.basename(self.unknown_snp_list))[0]
        tmp_file = osp.join(os.environ.get('DATA_IN'),str(snp_filename), 'known_variants_omni.pkl')
        tmp_file2 = osp.join(os.environ.get('DATA_IN'),str(snp_filename),'unknown_variants_omni.pkl')

        if os.path.exists(tmp_file):
            with open(tmp_file, 'rb') as f:
                known_variants = pickle.load(f)
            print("loaded cached version of variants from {}".format(tmp_file))

        if os.path.exists(tmp_file2):
            with open(tmp_file2, 'rb') as f:
                unknown_variants = pickle.load(f)
            print("loaded cached version of variants from {}".format(tmp_file2))
            known_variants.sort()
            unknown_variants.sort()
        else:
            known_variants, unknown_variants = read_predefined_variants(list(self.data.get_all_variant_ids()), self.marker_path, self.chm)
            known_variants.sort()
            print(f'known_variants:{known_variants}')
            os.makedirs(osp.join(os.environ.get('DATA_IN'),str(snp_filename)), exist_ok=True)
            with open(tmp_file, 'wb') as f:
                pickle.dump(known_variants, f)
            print("cached version of known_variants to {}".format(tmp_file))
            if self.unknown_snp_list is not None:
                snp_df = pd.read_csv(self.unknown_snp_list, index_col=1)
                snps = set(snp_df.index)
                #only consider snps that are present in the data
                snps = snps.intersection(set(self.data.get_all_variant_ids()))
                #get the indices for all snps in the microarray and the data
                snps = {self.data.get_variant_idx(x) for x in snps}
                print("{} Unkown variants total".format(len(unknown_variants)))
                unknown_variants = set(unknown_variants).intersection(snps)
                unknown_variants = list(unknown_variants)
                print("{} variants being trained".format(len(unknown_variants)))
            unknown_variants.sort()
            
            with open(tmp_file2, 'wb') as f:
                pickle.dump(unknown_variants, f)
            print("cached version of known_variants to {}".format(tmp_file2))
        j = DEFAULT_SUBWIN[self.dataset_iter]
        print(f"unknown variants before subsetting for sub-window size:{len(unknown_variants)}")
        unknown_variants=unknown_variants[j*self.subWin_size:(j+1)*self.subWin_size] # 87*100 = 8700
        print("{} variants being trained after sort ".format(len(unknown_variants)))
        print("{} unknown_variants being trained after sort ".format(unknown_variants))
        return known_variants, unknown_variants
    
    def flatten_tensor(self, x):
        """
        flatten the tensor
        """
        return x.flatten(dims=-1)#x[:,:,1]

    def process_genotype_array(self, genotype_array: np.ndarray, flatten: bool = False) -> torch.tensor:
        """
        Process a genotype array of arbitrary shape
        genotype_array: np array of shape [snps, samples, ploidy]
        returns: Tensor of shape [samples, snps]
        """
        genotype_tensor = torch.Tensor(genotype_array)
        genotype_tensor = genotype_tensor.permute(1, 0, 2)#[samples, snps, ploidy]
        if flatten:
            genotype_tensor = self.flatten_tensor(genotype_tensor)
        return genotype_tensor

    @timer
    def getClosestSnps(self, n_closest):
        self.closestSnpMatrixIdx=None
        if self.npt is not None:
            closestSnp_ls=[]
        for i, snp in enumerate(self.unknown_snp_ids):
            closest_per_snp = self.data.get_N_closest_idx(snp, n_closest=n_closest, known_snps=self.known_variants)
            if self.npt is not None: 
                closestSnp_ls.append(closest_per_snp)
            if i==0:
                self.closest_n_snps = closest_per_snp
            else:
                self.closest_n_snps = np.union1d(self.closest_n_snps, closest_per_snp)
            
        print(f"length of n closest snps:{self.closest_n_snps.shape}")
        #pick the middle n_closest snps on both sides
        if not self.old:
            self.closest_n_snps = np.concatenate((self.closest_n_snps[int(len(self.closest_n_snps)/2)-int(n_closest/2):int(len(self.closest_n_snps)/2)],
                            self.closest_n_snps[int(len(self.closest_n_snps)/2):int(len(self.closest_n_snps)/2)+int(n_closest/2):]))
        print(f"length of n closest snps after truncating:{self.closest_n_snps.shape}")
        if len(closestSnp_ls)>1:
            self.closestSnpMatrixIdx=torch.Tensor(np.vstack(closestSnp_ls))
    
    def getCombine(self):
        """
        combines in order the closest known snps selected and the unknown variants
        """
        self.closest_and_unknown_variants = list(self.closest_n_snps)
        self.closest_and_unknown_variants.extend(self.unknown_variants)
        self.closest_and_unknown_variants.sort()
        
class HaplotypeDataset(Dataset):
    @timer
    def __init__(self, haplotype, dataset_type,  model_name, transform_params, **kwargs):
        self.dataset_type=dataset_type
        self.model_name=model_name
        self.transform_params=transform_params
        # other kwargs option
        self.npt = kwargs.get('npt')
        self.Kmer = kwargs.get('Kmer')
        
        # initialize transformation object
        # self.transfromOption=modelSelect.get_selection()
        # can add more here, example granular_pop is not being used
        self.data = {'X':None, 'y':None, 'y_raw':None, 'auxQ':None}
        self.formData(haplotype)

    @timer
    def formData(self, haplotype):
        """
        this function forms the features and labels for each of train, valid, test
        dataset type is in ["train", "valid", "test"]
        model_name is string with params.model, example- 'subsetAttention'
        transform_params is a dict with params for any transformation, {win_size:win_size_value, win_stride:win_stride_value}
        """
        if self.dataset_type not in ["train", "valid", "test"]:
            raise ValueError
        
        # get the train, valid or test X and y depending on the dataset_type passed
        # form the features, 'X' 
        if self.dataset_type=='train':
            sample_idx = [haplotype.data.get_sample_idx(s_id) for s_id in haplotype.train_samples]
            self.data['y'] = haplotype.data.subset_genotypes(sample_ids = haplotype.train_samples.to_list(), variant_ids = [haplotype.data.get_variant_idx(var_id) for var_id in haplotype.unknown_snp_ids])
        elif self.dataset_type=='valid':
            sample_idx = [haplotype.data.get_sample_idx(s_id) for s_id in haplotype.val_samples]
            self.data['y'] = haplotype.data.subset_genotypes(sample_ids = haplotype.val_samples.to_list(), variant_ids = [haplotype.data.get_variant_idx(var_id) for var_id in haplotype.unknown_snp_ids])
        elif self.dataset_type=='test':
            sample_idx = [haplotype.data.get_sample_idx(s_id) for s_id in haplotype.test_samples]
            self.data['y'] = haplotype.data.subset_genotypes(sample_ids = haplotype.test_samples.to_list(), variant_ids = [haplotype.data.get_variant_idx(var_id) for var_id in haplotype.unknown_snp_ids])
        
        self.data['y'] = haplotype.process_genotype_array(self.data['y'])
        closest_x = haplotype.data.subset_genotypes(sample_ids= sample_idx, variant_ids=haplotype.closest_n_snps)
        closest_x = haplotype.process_genotype_array(closest_x)

        self.data['X'] = torch.cat(([closest_x[:,:,0], closest_x[:,:,1]]), dim=0)
        self.data['y'] = torch.cat(([self.data['y'][:,:, 0], self.data['y'][:,:, 1]]))

        if self.Kmer:
            self.data['X'] = torch.from_numpy(mapKmer(batchSeq=self.data['X'].cpu().detach().numpy(), win_size=self.transform_params['win_size'], win_stride=self.transform_params['win_stride']))
            self.data['y_raw'] = self.data['y']
            self.data['y'] = torch.from_numpy(mapKmer(batchSeq=self.data['y'].cpu().detach().numpy(), win_size=self.transform_params['win_size'], win_stride=self.transform_params['win_stride']))
            
        # apply the approporiate transform to X 
        print(f"features shape:{self.data['X'].shape}")
        self.data['X'] = self.data['X']
        print(f"features, labels :{self.data['X'].shape},{self.data['y'].shape}")
        if self.npt:
            self.data['auxQ'] = self.getAuxDataset(closest_x, haplotype.closestSnpMatrixIdx, haplotype.closest_n_snps)

    def getAllelCount(self):
        return self.data['y'].sum() if not self.Kmer else self.data['y_raw'].sum()

    @timer
    def _getNClosestMatrix(self, nClosestVal, closestSnpMatrixIdx, closest_n_snpsIdx):
        """
        given a combined n closest id and val vector, return the n closest per
        unknown snp id
        nClosestVal shape: [n_samplesxcombined_closest_snpsx2]
        """
        nClosestVal=torch.cat((nClosestVal[:,:,0], nClosestVal[:,:,1]),0) #n_samples*2, nCombinedSnps
        n_samples2=nClosestVal.shape[0] #n_samples*2
        nClosestMatrix = torch.zeros((n_samples2,closestSnpMatrixIdx.shape[0],closestSnpMatrixIdx.shape[1])) #n_samples*2, M, M'
        for i, v in enumerate(closest_n_snpsIdx):
            nClosestMatrix[:, torch.where(v==closestSnpMatrixIdx)[0],torch.where(v==closestSnpMatrixIdx)[1]]=nClosestVal[:,i][:,None]
        return nClosestMatrix

    def getAuxDataset(self, nClosestVal, closestSnpMatrixIdx, closest_n_snpsIdx):        
        nClosestMatrix=self._getNClosestMatrix(nClosestVal, closestSnpMatrixIdx, closest_n_snpsIdx)
        return nClosestMatrix

    def __len__(self):
        return len(self.data['X']) 

    def __getitem__(self, idx):
        X = self.data['X'][idx]
        y = self.data['y'][idx]
        if self.data['y_raw'] is not None:
            y_raw=self.data['y_raw'][idx]
            return X,(y,y_raw)
        if self.data['auxQ'] is not None:
            auxQ=self.data['auxQ'][idx]
            return X,(y,auxQ, idx)
        return X,y
            
class ImputationDataset(BaseDataset):
    def __init__(self, c):
        super(ImputationDataset, self).__init__(
            fixed_test_set_index=-c.fixed_test_set_index)   
        self.c = c
        print(f"printing all arguments:{self.c}")
        
    def load(self, dataset_iter, subWin_size):
        self.load_init=load_and_preprocess_imputation_dataset(self.c, dataset_iter, subWin_size)
        
        (self.data_table, self.N, self.D, self.cat_features, self.num_features,
            self.missing_matrix, self.data_table_y_raw, n_closest) = self.load_init()

        self.num_target_cols = []
        
        print(f"self.D, dataset_iter : {self.D},{dataset_iter}")
        if self.c.data_Kmer:
            subWin_size=int(subWin_size/self.c.Kmer_win_size)
            self.input_feature_dims = [] 
            if not self.c.old: assert self.D == subWin_size+int(n_closest/self.c.Kmer_win_size), "column length does not match"
        self.cat_target_cols = list(range(self.D-subWin_size,self.D))  # Binary classification
        self.is_data_loaded = True
        self.tmp_file_names = ['Imputation.csv']
        if self.c.load_interim_data:
            print(f"saving data for dataset_iter: {self.dataset_iter}")
            filepath=osp.join(os.environ.get('DATA_OUT'), 'npt', f'x{DEFAULT_WIN[self.dataset_iter]}')
            print(f"filepath:{filepath}")
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            np.save(''.join([filepath, '/data_table_', str(self.dataset_iter)]), self.data_table)
        
def getDataTable(train_dataset, val_dataset, test_dataset):
    """
    deprecated, not used, only used for experimentation
    """
    train_x = torch.cat((train_dataset.data['X'], train_dataset.data['X']))
    train_y = torch.cat((train_dataset.data['y'][:,0:300], train_dataset.data['y'][:,300:600]))

    val_x = torch.cat((val_dataset.data['X'], val_dataset.data['X']))
    val_y = torch.cat((val_dataset.data['y'][:,0:300], val_dataset.data['y'][:,300:600]))

    test_x = torch.cat((test_dataset.data['X'], test_dataset.data['X']))
    test_y = torch.cat((test_dataset.data['y'][:,0:300], test_dataset.data['y'][:,300:600]))

    data_table_X=torch.cat((train_x, val_x, test_x))
    data_table_y=torch.cat((train_y, val_y, test_y))
    
    data_table=torch.cat((data_table_X, data_table_y), dim=1)
    return data_table

class load_and_preprocess_imputation_dataset():
    def __init__(self, c, dataset_iter, subWin_size):
        """
        """
        data_dir = os.path.join(os.environ.get('DATA_IN'), "1k_genomes")
        meta_data_path = os.path.join(data_dir, '1k_genomes_sample_metadata.csv')
        # marker_path = os.path.join(os.environ.get('DATA_IN'), 'markers', 'InfiniumOmniExpress-24v1-2_A1-b38.strand')
        marker_path = os.path.join(os.environ.get('DATA_IN'), 'markers', 'HumanOmni2-5-8-v1-1-C-b37.strand')
        self.c = c
        self.dataset_iter = dataset_iter
        self.subWin_size = subWin_size
        if not os.path.exists(marker_path):
            raise Exception("Invalid marker path provided")
        print("Loading markers from {}".format(marker_path))
        # snp_list = os.path.join(os.environ.get('DATA_IN'), 'chm20_random_subsets', f'x{DEFAULT_WIN[self.dataset_iter]}.csv')
        snp_list = os.path.join(os.environ.get('CLEANED_CHR20_SNPS'), f'x{DEFAULT_WIN[self.dataset_iter]}')
        print(f"snp_list path:{snp_list}")
        vcf_filename="chr{}.1kg.phase3.v5a.vcf.gz"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"n_closest:{c.nClosest}")
        self.n_closest=int(c.nClosest*(1+c.incr_nClosest))
        print(f"n_closest = {self.n_closest}")
        self.haploData = Haplotype(self.n_closest, data_dir, vcf_filename, marker_path, snp_list, chromosome='20', npt=False, subWin_size=self.subWin_size, dataset_iter = self.dataset_iter, old=self.c.old)
        if c.incr_nClosest > 0:
            # sample q from n closest
            print(f"original n closest: {self.n_closest}, {self.haploData.closest_n_snps}")
            self.haploData.closest_n_snps = np.random.choice(self.haploData.closest_n_snps, int(c.sample_nClosest*self.n_closest), replace=False)
            self.haploData.closest_n_snps.sort()
            print(f" n closest after subsampling : {len(self.haploData.closest_n_snps)}, {self.haploData.closest_n_snps}")
        if c.debug_variants:
            print(f"saving variants for dataset_iter: {self.dataset_iter}")
            filepath=osp.join(os.environ.get('DATA_OUT'), 'npt', f'x{DEFAULT_WIN[self.dataset_iter]}',f'n_closest_{self.n_closest}',f'win_size_{self.subWin_size}')
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            np.save(''.join([filepath, '/knownVariants_', str(self.dataset_iter)]), self.haploData.closest_n_snps)
            np.save(''.join([filepath, '/unknownVariants_', str(self.dataset_iter)]), self.haploData.unknown_variants)
        self.train_dataset = HaplotypeDataset(self.haploData, "train", 'subsetAttention', {'win_size':c.Kmer_win_size, 'win_stride':c.Kmer_win_size}, npt=False, Kmer = c.data_Kmer)
        self.val_dataset = HaplotypeDataset(self.haploData, "valid", 'subsetAttention', {'win_size':c.Kmer_win_size, 'win_stride':c.Kmer_win_size}, npt=False, Kmer = c.data_Kmer)
        if not self.c.no_test_data: self.test_dataset = HaplotypeDataset(self.haploData, "test", 'subsetAttention', {'win_size':c.Kmer_win_size, 'win_stride':c.Kmer_win_size}, npt=False, Kmer = c.data_Kmer)
    
    def __call__(self):
        # use only a subset of the reference training proportion
        if self.c.train_proportion<1.0:
            num_train_samples_use = int(len(self.train_dataset.data['y'])*self.c.train_proportion)
            print(f"train samples being used:{num_train_samples_use}")
            print(f"before subsetting, number of train samples = {len(self.train_dataset.data['y'])}")
            self.train_dataset.data['X']=self.train_dataset.data['X'][:num_train_samples_use,:]
            self.train_dataset.data['y']=self.train_dataset.data['y'][:num_train_samples_use,:]
            print(f"after subsetting, number of train samples = {len(self.train_dataset.data['y'])}")

        if self.c.no_test_data: 
            data_table_y=torch.cat((self.train_dataset.data['y'], self.val_dataset.data['y']))
            data_table_X=torch.cat((self.train_dataset.data['X'], self.val_dataset.data['X']))
        else:
            data_table_y=torch.cat((self.train_dataset.data['y'], self.val_dataset.data['y'], self.test_dataset.data['y']))
            # tmp hack to switch between old and new
            if self.c.old:
                data_table_X=torch.cat((self.train_dataset.data['X'][:,:self.n_closest], self.val_dataset.data['X'][:,:self.n_closest], self.test_dataset.data['X'][:,:self.n_closest]))
            else:
                data_table_X=torch.cat((self.train_dataset.data['X'], self.val_dataset.data['X'], self.test_dataset.data['X']))
        if self.c.win_position:
            suffix_tensor=torch.tensor(DEFAULT_WINDOW_SUFFIX[self.dataset_iter]).float()[None,:]
            suffix_tensor=suffix_tensor.repeat(data_table_X.shape[0],1)
            data_table_X=torch.cat((data_table_X, suffix_tensor), dim=1)
        data_table_y_raw=None
        if self.c.data_Kmer:
            if self.c.no_test_data: 
                data_table_y_raw = torch.cat((self.train_dataset.data['y_raw'], self.val_dataset.data['y_raw']))
            else:
                data_table_y_raw = torch.cat((self.train_dataset.data['y_raw'], self.val_dataset.data['y_raw'], self.test_dataset.data['y_raw']))

        data_table=torch.cat((data_table_X, data_table_y), dim=1)
        data_table = data_table.cpu().detach().numpy()

        N, D = data_table.shape
        cat_features = list(range(0, D))
        num_features=[]
        print(f"shape of matrix:{N},{D}")
        missing_matrix = np.zeros((N, D), dtype=np.bool_)

        return data_table, N, D, cat_features, num_features, missing_matrix, data_table_y_raw, self.n_closest

