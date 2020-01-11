# UNet-Accuracy-Complexity
Pytorch implementation of the paper "Deep Neural Network for Multi-Organ Segmentation with Higher Accuracy and Lower Complexity"
# Code
The code implementation refers to https://github.com/jfzhang95/pytorch-deeplab-xception.  
The baseline U-Net architecture refers to https://github.com/milesial/Pytorch-UNet.  
Accuracy-Complexity Adjustment Module (ACAM) refers to https://github.com/d-li14/octconv.pytorch.  
Multi-scale Adjustable Module (MAM) refers to https://github.com/implus/SKNet.  
# Train
Change the path in mypath.py  

    CUDA_VISIBLE_DEVICES=0 python3 train.py --baseline unet2doct --lr 0.03 --workers 4 --epochs 100 --batch-size 4 --gpu-ids 0 --checkname journal --eval-interval 1 --dataset ctchest
