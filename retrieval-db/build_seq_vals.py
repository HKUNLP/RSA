import torch
import argparse
import numpy as np
import os
from tqdm import tqdm
import json

def ivecs_read(fname, count=-1, offset=0):
    a = np.fromfile(fname, dtype='int32', count=count, offset=offset)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname, count=-1, offset=0):
    return ivecs_read(fname, count=count, offset=offset).view('float32')




if __name__ == "__main__":
    need_seq = False    
    need_label = False 
    input_path = './pfam_db_dumps/pfam-A'
    output_path = 'pfam_seq.txt'
    label_path =  'pfam_label.txt'
    if need_seq:
        output_f = open(output_path, 'w')
    label_f = open(label_path, 'w')
    
    n_dumps = 18
    
    for dump in range(n_dumps):
        input_file = input_path + str(dump) + ".json"
        num_sentences = 0
        with open(input_file) as f_data:
            for line in f_data:
                line = line.strip()
                num_sentences += 1
        
        file_path = './pfam_vectors/vectors_dump_'+str(dump)+'.fvecs'
        keys = fvecs_read(file_path)
        
        assert keys.shape[0] == num_sentences
        with open(input_file) as f_in:
            for line in tqdm(f_in):
                line = line.strip()
                item = json.loads(line)
                if need_seq:
                    output_f.write(item['seq']+'\n')
                if need_label:
                    label_f.write(item['label']+'\n')
            
    '''
    
    input_path = 'pfam-small.json'
    output_path = 'pfam-small_seq.txt'
    output_f = open(output_path, 'w')
    
    with open(input_path) as input_f:
        for line in input_f:
            item = json.loads(line)
            output_f.write(item['seq']+'\n')
    '''