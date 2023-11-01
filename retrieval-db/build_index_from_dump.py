import argparse
import os
import numpy as np
import faiss
import time
from tqdm import tqdm


def ivecs_read(fname, count=-1, offset=0):
    a = np.fromfile(fname, dtype='int32', count=count, offset=offset)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname, count=-1, offset=0):
    return ivecs_read(fname, count=count, offset=offset).view('float32')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fvecs', type=str, help='memmap where keys and vals are stored')
    parser.add_argument('--dump_fvecs_dir', type=str, help='memmap where keys and vals are stored')
    parser.add_argument('--num_dumps', type=int, help='total number of dumps')
    parser.add_argument('--dimension', type=int, default=1280, help='Size of each key')
    #parser.add_argument('--dstore_fp16', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed for sampling the subset of vectors to train the cache')
    parser.add_argument('--ncentroids', type=int, default=4096, help='number of centroids faiss should learn')
    parser.add_argument('--code_size', type=int, default=64, help='size of quantized vectors')
    parser.add_argument('--probe', type=int, default=8, help='number of clusters to query')
    parser.add_argument('--faiss_index', type=str, help='file to write the faiss index')
    parser.add_argument('--num_keys_to_add_at_a_time', default=1000000, type=int,
                    help='can only load a certain amount of data to memory at a time.')
    parser.add_argument('--starting_id', type=int, help='dump index to start adding keys at')
    parser.add_argument('--device', type=int, help='device for training indexes')
    args = parser.parse_args()
    print(args)
    
    
    
    if not os.path.exists(args.faiss_index+".trained"):
        train_keys = fvecs_read(args.train_fvecs)   # key, value as (f(x_i),i)
    
        train_size = min(1000000, train_keys.shape[0])
        print("training database has {} samples".format(train_size))
        # Initialize faiss index
        
        quantizer = faiss.IndexFlatL2(args.dimension)
        index = faiss.IndexIVFPQ(quantizer, args.dimension,
            args.ncentroids, args.code_size, 8)
        index.nprobe = args.probe
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, args.device, index)
        gpu_index.verbose = True
        print('Training Index')
        
        np.random.seed(args.seed)
        random_sample = np.random.choice(np.arange(train_keys.shape[0]), size=[min(1000000, train_keys.shape[0])], replace=False)
        
        start = time.time()
        # Faiss does not handle adding keys in fp16 as of writing this.
        gpu_index.train(train_keys[random_sample].astype(np.float32))
        
        print('Training took {} s'.format(time.time() - start))

        index = faiss.index_gpu_to_cpu(gpu_index)
        print('Writing index after training')
        start = time.time()
        faiss.write_index(index, args.faiss_index+".trained")
        print('Writing index took {} s'.format(time.time()-start))
    
    print('Adding Keys from Dumps')
    index = faiss.read_index(args.faiss_index+".trained")  # continue loading
    start = 0
    start_time = time.time()
    end = 0
        
    
    for id in range(args.starting_id, args.num_dumps):
        start_time = time.time()
        file_path = args.dump_fvecs_dir + '/' + 'vectors_dump_'+str(id)+'.fvecs'
        keys = fvecs_read(file_path)
        print('loading features file took {} s'.format(time.time()-start_time))
        
        dump_start = 0
        dump_dstore_size = keys.shape[0]
        while dump_start < dump_dstore_size:
            dump_end = min(dump_start+args.num_keys_to_add_at_a_time, dump_dstore_size)
            end = start + dump_end - dump_start
            to_add = keys[dump_start:dump_end] #.copy()
            index.add_with_ids(to_add.astype(np.float32), np.arange(start, end))
            
            start = end
            dump_start += args.num_keys_to_add_at_a_time
            
            if (dump_start % 1000000) == 0:
                print('Added %d tokens so far' % start)
                print('Writing Index', start)
                faiss.write_index(index, args.faiss_index)
            

    print("Adding total %d keys" % (end))
    #print('Adding took {} s'.format(time.time() - start_time))
    print('Writing Index')
    start = time.time()
    faiss.write_index(index, args.faiss_index)
    print('Writing index took {} s'.format(time.time()-start))
        

    '''
    seq_embeds = fvecs_read('pfam-small.fvecs')
    seq_embeds = seq_embeds.repeat(100,1)
    index =  faiss.IndexFlatL2(1280)
    index.add(seq_embeds)
    D,I = index.search(seq_embeds[:1], 4) 
    faiss.write_index(index, 'pfam_small.index')
    
    n_keys, d = seq_embeds.shape
    quantizer = faiss.IndexFlatL2(d)
    cpu_index = faiss.IndexIVFPQ(quantizer, 1280, 2048, 32, 8, faiss.METRIC_INNER_PRODUCT)
    cpu_index.nprobe = 8
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res,6, cpu_index)
    gpu_index.verbose = True
    random_sample = np.random.choice(np.arange(n_keys), size=[min(200000, n_keys)], replace=False)
    gpu_index.train(seq_embeds[random_sample])
    trained_index = faiss.index_gpu_to_cpu(gpu_index)
    trained_index.verbose = True
    trained_index.add(seq_embeds)
    
    D,I = trained_index.search(seq_embeds[:1], 4) 
    print(I)
    '''