CUDA_VISIBLE_DEVICES=0 python3 train_s3fd.py --ngpu 1 --net resnet18 --save_folder ./weights/S3FD_RESNET18_200/ --num_workers 8 --batch_size 64 --pretrained ./weights/resnet18.pth
