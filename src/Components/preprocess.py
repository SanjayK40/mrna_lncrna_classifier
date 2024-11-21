from Components.data_loader import loader
import pandas as pd
import numpy as np

class process:
    def __init__(self,config):
        lncrna_path=config['Data_Path']['lncrna']
        mrna_path=config['Data_Path']['mrna']
        self.load=loader(lncrna_path=lncrna_path,mrna_path=mrna_path)

    def extract_kmers(self,sequence, k):
        kmers = []
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i + k]
            kmers.append(kmer)
        return kmers


    def create_kmer_matrix(self,fasta_data, all_kmers_set, k):
        num_sequences = len(fasta_data)
        num_unique_kmers = len(all_kmers_set)
        kmer_matrix = np.zeros((num_sequences, num_unique_kmers), dtype=int)

        for i, record in enumerate(fasta_data):
            sequence = str(record.seq).upper()
            kmers = self.extract_kmers(sequence, k)
            for j, kmer in enumerate(all_kmers_set):
                kmer_matrix[i, j] = kmers.count(kmer)
        return kmer_matrix


    def normalize(self,mRNA_kmer_matrix,lncRNA_kmer_matrix):
        max=mRNA_kmer_matrix.max() if mRNA_kmer_matrix.max()>lncRNA_kmer_matrix.max() else lncRNA_kmer_matrix.max()
        mRNA_kmer_matrix=mRNA_kmer_matrix/max
        lncRNA_kmer_matrix=lncRNA_kmer_matrix/max
        ones_column = np.ones((mRNA_kmer_matrix.shape[0], 1))
        zeros_column=np.zeros((lncRNA_kmer_matrix.shape[0],1))
        mRNA_kmer_matrix = np.hstack((mRNA_kmer_matrix, ones_column))
        lncRNA_kmer_matrix = np.hstack((lncRNA_kmer_matrix, zeros_column))
        df_mrna = pd.DataFrame(mRNA_kmer_matrix)
        df_lncRNA=pd.DataFrame(lncRNA_kmer_matrix)
        return df_mrna,df_lncRNA



    def preprocess(self,mRNA_data,lncRNA_data):    
     
        combined_sequences = [str(record.seq).upper() for record in mRNA_data] + [str(record.seq).upper() for record in lncRNA_data]
        
        all_kmers_set_1 = set()
        k = 1 
        all_kmers_set_2 = set()
        k = 2  
        all_kmers_set_3 = set()
        k = 3 
        all_kmers_set_4 = set()
        k = 4  
        all_kmers_set_5 = set()
        k = 5  

        for sequence in combined_sequences:
            kmers_1 = self.extract_kmers(sequence, 1)
            all_kmers_set_1.update(kmers_1)
            kmers_2 = self.extract_kmers(sequence, 2)
            all_kmers_set_2.update(kmers_2)
            kmers_3 = self.extract_kmers(sequence, 3)
            all_kmers_set_3.update(kmers_3)
            kmers_4 = self.extract_kmers(sequence, 4)
            all_kmers_set_4.update(kmers_4)
            kmers_5 = self.extract_kmers(sequence, 5)
            all_kmers_set_5.update(kmers_5)
            
            
        mRNA_kmer_matrix_1 = self.create_kmer_matrix(mRNA_data, all_kmers_set_1, 1)
        lncRNA_kmer_matrix_1 = self.create_kmer_matrix(lncRNA_data, all_kmers_set_1, 1)

        mRNA_kmer_matrix_2 = self.create_kmer_matrix(mRNA_data, all_kmers_set_2, 2)
        lncRNA_kmer_matrix_2 = self.create_kmer_matrix(lncRNA_data, all_kmers_set_2, 2)

        mRNA_kmer_matrix_3 = self.create_kmer_matrix(mRNA_data, all_kmers_set_3, 3)
        lncRNA_kmer_matrix_3 = self.create_kmer_matrix(lncRNA_data, all_kmers_set_3, 3)

        mRNA_kmer_matrix_4 = self.create_kmer_matrix(mRNA_data, all_kmers_set_4, 4)
        lncRNA_kmer_matrix_4 = self.create_kmer_matrix(lncRNA_data, all_kmers_set_4, 4)

        mRNA_kmer_matrix_5 = self.create_kmer_matrix(mRNA_data, all_kmers_set_5, 5)
        lncRNA_kmer_matrix_5 = self.create_kmer_matrix(lncRNA_data, all_kmers_set_5, 5)
        df_mrna_1,df_lncRNA_1 = self.normalize(mRNA_kmer_matrix_1,lncRNA_kmer_matrix_1)
        df_mrna_2,df_lncRNA_2 = self.normalize(mRNA_kmer_matrix_2,lncRNA_kmer_matrix_2)
        df_mrna_3,df_lncRNA_3 = self.normalize(mRNA_kmer_matrix_3,lncRNA_kmer_matrix_3)
        df_mrna_4,df_lncRNA_4 = self.normalize(mRNA_kmer_matrix_4,lncRNA_kmer_matrix_4)
        df_mrna_5,df_lncRNA_5 = self.normalize(mRNA_kmer_matrix_5,lncRNA_kmer_matrix_5)
        
        df1=pd.concat([df_mrna_1, df_lncRNA_1], ignore_index=True)
        df2=pd.concat([df_mrna_2, df_lncRNA_2], ignore_index=True)
        df3=pd.concat([df_mrna_3, df_lncRNA_3], ignore_index=True)
        df4=pd.concat([df_mrna_4, df_lncRNA_4], ignore_index=True)
        df5=pd.concat([df_mrna_5, df_lncRNA_5], ignore_index=True)
        
        return df1,df2,df3,df4,df5
