#!/bin/bash
source /home/rr568/NPT/non-parametric-transformers/scripts/init_env.sh

echo 'Begin npt'
free -h
python $USER_PATH/run.py --data_set synthetic_att --model_dim_hidden 32 --model_num_heads 1 --model_stacking_depth 1 --data_path /share/kuleshov/richras/npt/data/spin_runs --data_force_reload True --exp_descr SPIN_synthetic_ncols_4_p_0.1_2_cent_mixed_labels_rev_0.1_0.1_I_2 --exp_batch_size -1 --model_hidden_dropout_prob 0.4 --viz_att_maps False --model_checkpoint_key perceiver_F_mw --exp_num_total_steps 15000 --np_seed 42 --perceiver_D True --project Synthetic --startAttBlock ABD --exp_eval_every_n 1 --model_bert_augmentation True --model_augmentation_bert_mask_prob "dict(train=0, val=0, test=0)" --dataSynthetic_query 600 --dataSynthetic_n_rows 6000 --dataSynthetic_n_cols 4 --allow_even_only_stacking_depth False --bernoulliNoise 0.1 --ABLA False --num_inds_I 2

echo 'Submission finished'
