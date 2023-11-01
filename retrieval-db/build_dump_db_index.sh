# need one 32G gpu
python build_index_from_dump.py --device 0 --train_fvecs pfam-small.fvecs --dump_fvecs_dir pfam_vectors --num_dumps 18 --dimension 1280 \
    --faiss_index pfam.index --num_keys_to_add_at_a_time 100000 --starting_id 0 --code_size 32