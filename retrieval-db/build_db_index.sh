# need one 32G gpu
python build_index.py --device 6 --dstore_fvecs pfam-small.fvecs --dstore_size 100000000 --dimension 1280 \
    --faiss_index pfam-small.index --num_keys_to_add_at_a_time 100000 --starting_point 0 --code_size 32