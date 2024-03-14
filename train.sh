# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29000 --num_processes 8 --mixed_precision 'fp16' train.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29000 --num_processes 8  train.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29000 --num_processes 8  preprocess.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py --optim adamw --learning_rate 0.001 --weight_decay 0.0005 --batch_size 8 --epochs 8 --work_dir exp/adamw_lr_1e-3_wd_5e-4_bs_8_epoch_8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py --optim adamw --learning_rate 0.001 --weight_decay 0.0005 --batch_size 8 --epochs 8 --ckpt ckpts/yolov8l-world.pt --work_dir exp/yolov8l_adamw_lr_1e-3_wd_5e-4_bs_8_epoch_8
python model_aggregation.py --ckpt exp/adamw_lr_1e-3_wd_5e-4_bs_8_epoch_8/ckpts/iter_final.pth --save_model_path weights --save_model_name adamw_lr_1e-3_wd_5e-4_bs_8_epoch_8.pth
python model_aggregation.py --ckpt exp/yolov8l_adamw_lr_1e-3_wd_5e-4_bs_8_epoch_8/ckpts/iter_50000.pth --save_model_path weights --save_model_name yolov8l_adamw_lr_1e-3_wd_5e-4_bs_8_epoch_8.pth


