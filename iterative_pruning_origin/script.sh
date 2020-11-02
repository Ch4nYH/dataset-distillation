python -u main_imp_std.py --data ../data \
    --dataset cifar10 --task supervised \
    --seed 1 \
    --save_dir LT_cifar10 \
    --pruning_times 15 \
    --rate 0.2 \
    --prune_type lt 
