from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
class loader:
    def __init__(self,mrna_path=None,lncrna_path=None):
        self.mPath=mrna_path
        self.lPath=lncrna_path
    def load(self):
        mRNA_data = list(SeqIO.parse(self.mPath, 'fasta'))
        lncRNA_data = list(SeqIO.parse(self.lPath, 'fasta'))
        return mRNA_data,lncRNA_data
# if __name__=='__main__':
#     lncrna_path=config['Data_Path']['lncrna']
#     mrna_path=config['Data_Path']['mrna']
#     load=loader(mrna_path=mrna_path,lncrna_path=lncrna_path)
#     mrna_data,lncrna_data=load.load()
#     # print(mrna_data,lncrna_data)