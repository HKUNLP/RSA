import numpy as np
import os
from tqdm import tqdm

def ivecs_read(fname, count=-1, offset=0):
    a = np.fromfile(fname, dtype='int32', count=count, offset=offset)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname, count=-1, offset=0):
    return ivecs_read(fname, count=count, offset=offset).view('float32')

def ivecs_write(fname, m):
    n, d = m.shape
    #print("dimension is :"+str(d))
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    #print(m1.shape)
    m1.tofile(fname)
    
def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))

if __name__ == "__main__":
    vectors_dir = 'pfam_vectors'
    all_files = os.listdir(vectors_dir)
    vector_fvec = 'pfam-all.fvecs'
    
    for id in tqdm(range(len(all_files))):
        file_path = vectors_dir+'/vectors_dump_'+str(id)+'.fvecs'
        features = fvecs_read(file_path)
        
        fvecs_write(vector_fvec, features)