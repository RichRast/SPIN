# This code is used from https://github.com/emarro/deep_pop_gen/blob/main/src/vcf_processor.py
# to process raw vcf files from 1KG 

from numpy.core.fromnumeric import var
import allel
import numpy as np
from typing import List, Tuple, Callable
import pandas as pd
from operator import itemgetter
import random
import os

def random_variants(num_variants: int, array_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    generate indices of the variants that are known and unknown
    Parameters:
        array_size[int]: the nuber of markers to sample
    returns:
        Tuple  ofindicies of the known and unknown variants
    """
    selected_variants = np.random.choice(num_variants, array_size, replace=False)#randomly select the genotyped markers
    unknown_variants = np.setdiff1d(np.arange(num_variants), selected_variants)#Generate indices of unkown markers
    return (selected_variants, unknown_variants)

def split_test_train(known: np.ndarray, unknown: np.ndarray, num_test: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split the known and unkown array into test and train sets
    """
    _, num_samples, _ = known.shape
    test_samples, train_samples = random_variants(num_samples, num_test)
    known_test = known[:, test_samples, :]
    known_train = known[:, train_samples, :]
    unknown_test = unknown[:, test_samples, :]
    unknown_train = unknown[:, train_samples, :]
    return (np.array([known_train, unknown_train]), np.array([known_test, unknown_test]))



def read_predefined_variants(variants: List[str], markers: str, chromosome: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read the indicies of the known variants from file [markers] and 
    return the known and unkown indicies given [variants]

    Expects markers to be path to a .strand file(https://www.well.ox.ac.uk/~wrayner/strand/)

    Parameters:
        markers[str]: path to file of rsids (seperated by new lines)
        variants[list]: 1d array of the strings of the variants for a given vcf file
    returns:
        Tuple  ofindicies of the known and unknown variants
    """
    #Process variants ids from markers
    df = pd.read_csv(markers, sep='\t', header=None, names = ['id', 'chrom', 'pos', 'perc', 'strand', 'TAG'])
    df = df.loc[df["chrom"]==chromosome]
    known_variants = df["id"].values
    """
    with open(markers, 'r') as f:
        known_variants = f.read().split('\n')
        known_variants = [x.split("\t")[0] for x in known_variants if x.split("\t")[1] == chromosome]
    print(known_variants)
    """


    #find the index of every known variant
    selected_variants = list()
    for variant in known_variants:
        #TODO: add logging for if variant id isn't found in queried vcf file
        try:
            variant_id = variants.index(variant)
            selected_variants.append(variant_id)
        except ValueError:
            pass
    selected_variants = np.array(selected_variants)

    #get the indicies of all the uknown variants
    unknown_variants = np.setdiff1d(np.arange(len(variants)), selected_variants)#Generate indices of unkown markers
    return (selected_variants, unknown_variants)
def generate_random_samples(sample_metadata: str, n: int, samples: np.ndarray) -> pd.Series:
    """
    Generate random samples for testing by sampling n samples per population as defiend in sample_metadata.
    Only samples from ids given by [samples].
    arguments:
        sample_metadata[str] - path to sample metadata
        n[int] - number of samples per population
        samples[np.ndarray[str]] - array of samples present in data
    """
    sample_df = pd.read_csv(sample_metadata)
    #Make sure we only look at samples we actually have data for
    sample_df = sample_df[sample_df['Sample'].isin(samples)]
    #The samples in the test set
    test_samples = pd.Series()
    #For every pop, select samples to be part of test set
    for pop in sample_df['Population'].unique():
        pop_samples = sample_df['Sample'][sample_df['Population'] == pop].sample(n=n) # sample 2 from every pop
        test_samples = pd.concat([test_samples, pop_samples])
    return test_samples

def split_samples(samples: pd.Series, test_samples: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Generate two boolean arrays from samples denoting whether the each element of [samples] is in the ref or
    test set
    arguments:
        samples[pd.Series[str]] - the array of sample ids to split
        test_samples[pd.Series[str]] - the array of samples in the test set
    returns:
        test_membership[pd.Series[bool]] - boolean series of whether the samples are in test
        ref_membership[pd.Series[bool]] - negation of [test_membership]
    """
    test_membership = samples.isin(test_samples)

    ref_membership = ~test_membership
    return test_membership, ref_membership 
    
class VCF_Loader:
    def __init__(self, vcf_file: str, chromosome: str, start_pos: int, end_pos: int=None) -> None:
        """
        Load in a VCF file for use in a ml model
        """
        data = allel.read_vcf(vcf_file, region=chromosome)#, exclude_fields=['CHROM', 'FILTER_PASS','QUAL'])
        print(data.keys())
        print(data)
        window = data['variants/POS'] > start_pos
        if end_pos is not None:
            window = window & (data['variants/POS'] < end_pos)
        self.samples = data['samples']#List of samples ids
        self.alts = data['variants/ALT'][window]
        self.pos = data['variants/POS'][window]
        self.ref = data['variants/REF'][window]
        self.variants = data['variants/ID'][window]#list of variant ids
        self.genotypes = data['calldata/GT'][window,:,:]#np array (num_variants, num_samples, ploidy=2)

        self.sample_to_index = {sample: idx for idx, sample in enumerate(self.samples)}
        self.variant_to_index = {variant: idx for idx, variant in enumerate(self.variants)}

    def get_sample_idx(self, sample_id: str) -> int:
        """
        Get the index for this sample
        """
        try:
            return self.sample_to_index[sample_id]
        except:
            return -1

    def get_variant_idx(self, variant_id: str) -> int:
        """
        Get the index for this variant
        """
        try:
            return self.variant_to_index[variant_id]
        except: 
            return -1
        

    def get_sample_genotypes(self, sample_id: str) -> np.ndarray:
        """
        Look up the genotype for the given sample
        """
        return self.genotypes[:, self.get_sample_idx(sample_id), :]

    def get_samples_genotypes(self, sample_ids: List[str]) -> np.ndarray:
        """
        Lok up genotyps for all variants for given sample
        """
        return self.genotypes[:,[self.get_sample_idx(s_id) for s_id in sample_ids],:]

    def get_variant_genotypes(self, variant_id: str) -> np.ndarray:
        """
        Look up the genotype for the given variant
        """
        return self.genotypes[self.get_variant_idx(variant_id), :, :]

    def get_variants_genotypes(self, variant_ids: List[str]) -> np.ndarray:
        """
        Lok up genotyps for all variants for given variant
        """
        return self.genotypes[[self.get_variant_idx(v_id) for v_id in variant_ids],:, :]
    
    def get_all_genotypes(self) -> np.ndarray:
        """
        Return the genotypes for every variant and samples
        [num_variants, num_samples, 2]
        """
        return self.genotypes




class VCF_Processing:
    def __init__(self, vcf_file: str, chromosome: str,  alt_number: int=6) -> None:
        """
        arguments:
            vcf_file[str] - path to the vcf file to process
            chromosome[str] - the chromosome we're predicting on
            alt_number[int] - the number of alternte alles to allow per variant
        """
        #read the data
        data = allel.read_vcf(vcf_file, region=chromosome, alt_number=alt_number, tabix=os.path.join(os.environ.get('DATA_IN'),'1k_genomes/chr20.1kg.phase3.v5a.vcf.gz.tbi'))
        # data = allel.read_vcf(vcf_file)
        #all data
        self.samples = data['samples']#List of samples ids
        self.alts = data['variants/ALT']
        self.chroms = data['variants/CHROM']
        self.filter = data['variants/FILTER_PASS']
        self.pos = data['variants/POS']
        self.qual = data['variants/QUAL']
        self.ref = data['variants/REF']
        #Load in the genotyping data
        self.variants = data['variants/ID']#list of variant ids
        self.genotypes = data['calldata/GT']#np array (num_variants, num_samples, ploidy=2)

    def get_samples(self) -> np.ndarray:
        """
        Return string array of samples in this vcf file
        """
        return self.samples
    
    def get_all_variant_ids(self) -> np.ndarray:
        """
        Return numpy array of strings of all variant ids
        """
        return self.variants
    
    def get_genotypes(self, variants: np.ndarray = None) -> np.ndarray:
        """
        Return the genotypes for the variants specified by [variants], return all if None
        """
        if variants is None:
            return self.genotypes
        return self.genotypes[variants,:,:]

    def write_data(self, file_name: str, sample_membership: List[bool], selected_variants: np.ndarray = None, flatten: bool = False) -> None:
        """
        Write the inputs and outputs for this vcf file to file_name
        Parameters:
            file_name[str] - the path to write the vcf file to
            sample_membership[list[bool]] - boolean list of  which samples to write
            selected_variants[np.ndarray] - array if indices of variants to include
        """
        #write metadata 
        #write CHROM POS ID REF ALT QUAL FILTER INFO FORMAT samples.... for every variant
        if selected_variants is None:
            selected_variants = np.arange(0, self.genotypes.shape[0])
        def process_id(ids: str, version: int) -> str:
            """
            Process ids that correspond to the same position when multiple ids map to one position
            """
            if ";" in ids:
                ids = ids.split(";")
                return ";".join([x+"v{}".format(version) for x in ids])
            if len(ids) == 0:
                ids = ''.join(random.choices('ABCDEFGHIJKLMNPQRS', k=9))
            return ids + "v{}".format(version)
        def write_line(chrom: str, pos: int, id: str, ref: str, alts: List[str],
                qual: str, filter: str, genotypes: np.ndarray, flatten: bool) -> str:
            """
            Generate the string for the variant specified by arguments
            The string should be one self contained line of the vcf file
            """
            if flatten:
                s = ''
                for idx, alt_base in enumerate([x for x in alts if x != ""]):
                    s += "{}\t{}\t{}\t{}\t{}\t{}\t{}\t.\tGT\t".format(
                        chrom, pos, process_id(id, idx), ref, alt_base, qual, filter) + \
                        "\t".join([str(int(x[0] != 0)) + "|" + str(int(x[1] != 0)) for x in genotypes]) + "\n"
                return s


            return "{}\t{}\t{}\t{}\t{}\t{}\t{}\t.\tGT\t".format(
                    chrom, pos, id, ref, ",".join([x for x in alts if x != ""]), qual, filter) + \
                    "\t".join([str(x[0]) + "|" + str(x[1]) for x in genotypes]) + "\n"



        alts = self.alts[selected_variants,:]
        chroms = self.chroms[selected_variants]
        filters = self.filter[selected_variants]
        positions = self.pos[selected_variants]
        quals = self.qual[selected_variants]
        refs = self.ref[selected_variants]
        variants = self.variants[selected_variants]
        samples = self.samples[sample_membership]
        genotypes = self.genotypes[selected_variants, :, :]
        genotypes = genotypes[:, sample_membership, :]


        with open(file_name, 'w') as f:
            #Write header
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" +
                    "\t".join(samples) + "\n")
            for chrom, pos, id, ref, alt, qual, filter, data in sorted(zip(chroms, positions, variants,refs, alts, quals, filters, genotypes),key=itemgetter(1)):
                f.write(write_line(chrom, pos, id, ref, alt, qual, filter, data, flatten))
        print("Done writing {}".format(file_name))
    
    def calc_non_reference(self) -> np.ndarray:
        """
        Calculate the percentage of non-reference allels for each variant(aggregate for multiallelic sites)
        """
        mac = self.calc_mac()
        num_genotypes = (self.genotypes.shape[1]*2)
        return mac/num_genotypes
    
    def calc_mac(self) -> np.ndarray:
        """
        Calculate the minor allele count for every variant 
        """
        mac = (((np.where(self.genotypes == 0, self.genotypes, 1)).sum(axis=-1)).sum(axis=-1))
        return mac


    def select_variants(self, cutoff: float) -> np.ndarray:
        """
        Generate a boolean array of variants that are above the cutoff.
        arguments:
            cutoff[float] - the minor allele cutoff. If cutoff < 1.0 assume it's minor allel freq
                            otherwise(cutoff >= 1.0) assume minor allele count.
        """
        if cutoff < 1.0: #Assume we were given a minor allel frequency(if maf==1.0 no need to filter)
            allele_freqs = self.calc_non_reference()
            keep = allele_freqs >= cutoff
        else: #cutoff >= 1.0, assume we are cutting based off of the minor allele count
            allele_counts = self.calc_mac()
            keep = allele_counts >= cutoff
        return keep
    
    def filter_variants(self, filtered_variants: np.ndarray) -> None:
        """
        Inplace filtering of all variants in this VCF file by given variant array
        """
        self.alts = self.alts[filtered_variants,:]
        self.chroms = self.chroms[filtered_variants]
        self.filter = self.filter[filtered_variants]
        self.pos = self.pos[filtered_variants]
        self.qual = self.qual[filtered_variants]
        self.ref = self.ref[filtered_variants]
        self.variants = self.variants[filtered_variants]

        self.genotypes = self.genotypes[filtered_variants, :, :]
    
    def build_indicies(self) -> None:
        """
        Construct mapping indicies for variants and for samples
        """
        self.sample_to_index = {sample: idx for idx, sample in enumerate(self.samples)}
        self.variant_to_index = {variant: idx for idx, variant in enumerate(self.variants)}
    
    def filter_windows(self, start_pos: int, end_pos: int=None ) -> np.ndarray:
        window = self.pos > start_pos
        if end_pos is not None:
            window = window & (self.pos < end_pos)
        return window
    
    def get_N_closest(self, snp_id: str, n_closest: int, known_snps:np.ndarray):
        """
        Get the n_closest snps to snp_id

        returns snp_ids

        Precondition: n_closest < total number of snps
        """
        closest = []
        snp_idx = self.get_variant_idx(snp_id) #with indices built, constant time lookup
        snp_pos = self.pos[snp_idx] #constant time
        distances = np.abs(self.pos[known_snps] - snp_pos) #take the closest known N regardless of orientation
        id_dist_pairings = zip(self.variants[known_snps], distances)
        id_dist_pairings = iter(sorted(id_dist_pairings, key=lambda x: x[1]))
        while len(closest) < n_closest:#constant time wrt n_closest
            try:
                variant_id, distance = next(id_dist_pairings)
                if distance > 0:#possible edge case where there are (same snp diff name) or (same pos diff allele)
                    #remove edge case from consideration
                    closest.append(variant_id)
            except StopIteration:
                #More useful error logging
                print("Error processing snp %s exiting with only %i/%i closest snps" %(snp_id, len(closest), n_closest))
                return closest
        return closest

    def get_N_closest_idx(self, snp_id: str, n_closest: int, known_snps:np.ndarray):
        """
        Get the n_closest snps to snp_id

        returns the indicies of the N closest snps instead of the names.

        Precondition: n_closest < total number of snps
        """
        closest = []
        snp_idx = self.get_variant_idx(snp_id)
        snp_pos = self.pos[snp_idx]
        distances = np.abs(self.pos[known_snps] - snp_pos) #take the closest known N regardless of orientation
        id_dist_pairings = zip(self.variants[known_snps], distances)
        id_dist_pairings = iter(sorted(id_dist_pairings, key=lambda x: x[1]))
        while len(closest) < n_closest:
            try:
                variant_id, distance = next(id_dist_pairings)
                if distance > 0:
                    closest.append(self.get_variant_idx(variant_id))
            except StopIteration:
                print("Error processing snp %s exiting with only %i/%i closest snps" %(snp_id, len(closest), n_closest))
                return closest
        return closest
    
    def snp_distance(self, ref_snp_id: str, alt_snp_id: str):
        """
        Get the abs(distance) between the two provided snp ids in 
        base pairs. 
        """
        ref_snp_idx = self.get_variant_idx(ref_snp_id)
        ref_snp_pos = self.pos[ref_snp_idx]
        alt_snp_idx = self.get_variant_idx(alt_snp_id)
        alt_snp_pos = self.pos[alt_snp_idx]
        return abs(ref_snp_pos - alt_snp_pos)
 
    def get_sample_idx(self, sample_id: str) -> int:
        """
        Get the index for this sample
        """
        return self.sample_to_index[sample_id]

    def get_variant_idx(self, variant_id: str) -> int:
        """
        Get the index for this variant
        """
        return self.variant_to_index[variant_id]
        

    def get_sample_genotypes(self, sample_id: str) -> np.ndarray:
        """
        Look up the genotype for the given sample
        """
        return self.genotypes[:, self.get_sample_idx(sample_id), :]

    def get_samples_genotypes(self, sample_ids: List[str]) -> np.ndarray:
        """
        Lok up genotypes for all variants for given sample
        """
        return self.genotypes[:,[self.get_sample_idx(s_id) for s_id in sample_ids],:]

    def get_variant_genotypes(self, variant_id: str) -> np.ndarray:
        """
        Look up the genotype for the given variant
        """
        return self.genotypes[self.get_variant_idx(variant_id), :, :]

    def get_variants_genotypes(self, variant_ids: List[str]) -> np.ndarray:
        """
        Lok up genotypes for all samples for given variant
        """
        return self.genotypes[[self.get_variant_idx(v_id) for v_id in variant_ids],:, :]
    
    def get_all_genotypes(self) -> np.ndarray:
        """
        Return the genotypes for every variant and samples
        [num_variants, num_samples, 2]
        """
        return self.genotypes
    
    def subset_genotypes(self, sample_ids: List[str], variant_ids: List[str]) -> np.ndarray:
        """
        Return the subset of genotypes specified by arguments
        """
        if type(variant_ids[0]) is str:
            variant_ids = [self.get_variant_idx(x) for x in variant_ids]
        d =  self.genotypes[variant_ids,:, :]

        if type(sample_ids[0]) is str:
            sample_ids = [self.get_sample_idx(s_id) for s_id in sample_ids]

        d =  d[:,sample_ids,:]
        d =  np.where(d == 0, d, 1)
        return d


