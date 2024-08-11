python train.py --task 'prob' --model 'PolarGate' --device 0 --batch_size 256 --eval_step 1 \
       --split_file 0.05-0.05-0.9 --layer_num 9 --feature_type 'one-hot' --in_dim 3

python train.py --task 'tt' --model 'PolarGate' --device 0 --batch_size 256 --eval_step 1 \
       --split_file 0.05-0.05-0.9 --layer_num 9 --feature_type 'one-hot' --in_dim 3