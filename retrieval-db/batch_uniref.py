from Bio.SeqIO.FastaIO import SimpleFastaParser
import json
from tqdm import tqdm
import random
import pickle
import os

def edit_distance(string1, string2):
    if len(string1) > len(string2):
        difference = len(string1) - len(string2)
        string1_temp = string1[:len(string2)]
        string2_temp = string2

    elif len(string2) > len(string1):
        difference = len(string2) - len(string1)
        string2_temp = string2[:len(string1)]
        string1_temp = string1

    else:
        string2_temp = string2
        string1_temp = string1
        difference = 0
    
    

    for i in range(len(string1_temp)):
        if string1_temp[i] != string2_temp[i]:
            difference += 1
    return difference

if __name__ == "__main__":
    all_seq = set([])
    
    if not os.path.exists('uniref_segs.pickle'):
        with open("uniref50.fasta") as handle:
            for values in tqdm(SimpleFastaParser(handle)):
                sequence = values[1]
                ind = 0
                while ind < len(sequence):
                    new_ind = min(ind+200, len(sequence))
                    all_seq.add(sequence[ind:new_ind])
                    ind += 200
        
    print("total "+str(len(all_seq))+" sequences")
        
    
    if not os.path.exists('uniref_db_dumps'):
        os.mkdir('uniref_db_dumps')
        
    num_dumps = 20
    all_f = dict([])
    for i in range(num_dumps):
        all_f[i] = open('uniref_db_dumps/uniref_'+str(i)+'.json', 'w')
    
    i=0
    for item in tqdm(all_seq):
        i = i+1
        dump = i % num_dumps
        json_item = {'label': i, 'seq':item}
        all_f[dump].write(json.dumps(json_item)+'\n')
    