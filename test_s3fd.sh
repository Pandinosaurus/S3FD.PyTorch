# env

# export PATH="/data/anaconda3/bin:$PATH"
# export TORCH_HOME='/data/.torch/'

# cd /data/task/Detection/dev/FaceBoxes.PyTorch-master

CUDA_VISIBLE_DEVICES=0 python3 test_s3fd_wider_mv2.py --trained_model weights/S3FD_FairNAS_B/Final_FairNAS_B_S3FD.pth  --save_folder eval/S3FD_FairNAS_B
