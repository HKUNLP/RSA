import torch
import argparse
import numpy as np
import os
from tqdm import tqdm
import json
import esm

#model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
#batch_converter = alphabet.get_batch_converter()

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

def main(args, num_dump, batch_size=124, device='cuda:0'):
    
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    
    model.eval()
    
    model = model.to(device)
    print("DUMP: ", num_dump)

    if args.file_path is not None:
        input_file = args.file_path
        save_file = args.save_path
    else:
        data_file = "./pfam_db_dumps/pfam-A"
        save_dir = "./pfam_vectors/"
        save_file = save_dir + "vectors_dump_"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            
        input_file = data_file + str(num_dump) + ".json"
        save_file = save_file + str(num_dump) + ".fvecs"

    num_sentences = 0
    with open(input_file) as f_data:
        for line in f_data:
            line = line.strip()
            num_sentences += 1

    sentence_batch = []
    with open(save_file, "wb") as f_out:
        with open(input_file) as f_in:
            for line in tqdm(f_in, total=num_sentences):
                line = line.strip()
                item = json.loads(line)
                sentence_batch.append((item['label'], item['seq']))
                if len(sentence_batch) == batch_size:
                    batch_labels, batch_strs, batch_tokens = batch_converter(sentence_batch)
                    b, n = batch_tokens.shape
                    if n>1024:
                        batch_tokens = batch_tokens[:,:1024]
                    batch_tokens = batch_tokens.to(device)
                    with torch.no_grad():
                        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                        token_representations = results["representations"][33]
                    sequence_representations = []
                    for i, (_, seq) in enumerate(sentence_batch):
                        sequence_representations.append(np.array(token_representations[i, 1 : len(seq) + 1].mean(0).cpu()))
                    fvecs_write(f_out, np.array(sequence_representations))
                    sentence_batch = []

            if len(sentence_batch) != 0:
                batch_labels, batch_strs, batch_tokens = batch_converter(sentence_batch)
                b, n = batch_tokens.shape
                if n>5:
                    batch_tokens = batch_tokens[:,:5]
                batch_tokens = batch_tokens.to(device)
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                    token_representations = results["representations"][33]
                sequence_representations = []
                for i, (_, seq) in enumerate(sentence_batch):
                    sequence_representations.append(np.array(token_representations[i, 1 : len(seq) + 1].mean(0)))
                fvecs_write(f_out, np.array(sequence_representations))
                sentence_batch = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump",
                        default=0,
                        type=int,
                        required=True,
                        help="Which dump should be processed")
    parser.add_argument("--batch_size",
                        default=1,
                        type=int,
                        help="Batch size")
    parser.add_argument("--device",
                        default=0,
                        type=int,
                        help="device")
    parser.add_argument("--file_path",
                        default=None,
                        type=str,
                        required=False,
                        help="use your own file")
    parser.add_argument("--save_path",
                        default=None,
                        type=str,
                        required=False,
                        help="use your own file")
    
    args = parser.parse_args()
    main(args, args.dump, args.batch_size, 'cuda:'+str(args.device))








