# Note: do not forget to change MODEL_FLAGS if other pretrained DDPMs are used.

MODEL_FLAGS="--attention_resolutions 32,16,8 
            --class_cond False 
            --diffusion_steps 1000 
            --dropout 0.1 
            --image_size 256 
            --learn_sigma True 
            --noise_schedule linear 
            --num_channels 256 
            --num_head_channels 64 
            --num_res_blocks 2 
            --resblock_updown True 
            --use_fp16 True 
            --use_scale_shift_norm True"
            
DATASET=$1  # Available datasets: bedroom_28, ffhq_34, cat_15, horse_21, celeba_19, ade_bedroom_30
NAME=$2     # Result directory
DEVICE=$3   # GPU Num

# CUDA_VISIBLE_DEVICES (사용할 GPU 번호 설정)
CUDA_VISIBLE_DEVICES=${DEVICE} python train_etf.py --exp experiments/${DATASET}/etf.json $MODEL_FLAGS --name ${NAME}