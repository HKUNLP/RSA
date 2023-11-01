1. Batch pfam sequence data with batch_pfam.py for concurrent building vectors.
2. Extract features of the sequences: 
    ```
        python extract_features.py --dump dump_id --device gpu_id 
    ```
3. run build_dump_db_index.sh, which trains a faiss index and add new vectors into it.
4. run build_seq_vals.py to combine the batches into a full file of sequence values and labels