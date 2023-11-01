from Bio.SeqIO.FastaIO import SimpleFastaParser
import json
from tqdm import tqdm
import random
if __name__ == "__main__":
    all_seq = []
    with open("Pfam-A.fasta") as handle:
        for values in SimpleFastaParser(handle):
            all_seq.append(values)
    print("total "+str(len(all_seq))+" sequences")
    
    
    num_dumps = 18
    all_f = dict([])
    for i in range(num_dumps):
        all_f[i] = open('pfam_db_dumps/pfam-A'+str(i)+'.json', 'w')
    
    for item in tqdm(all_seq):
        i = i+1
        dump = i % num_dumps
        json_item = {'label': item[0], 'seq':item[1]}
        all_f[dump].write(json.dumps(json_item)+'\n')
        
