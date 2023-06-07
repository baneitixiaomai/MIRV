# Mutual Information Regularization for Weakly-supervised RGB-D Salient Object Detection (TCSVT2023)
![](https://github.com/baneitixiaomai/MIRV/blob/main/overview_00.png)  
 [[paper]](https://arxiv.org/pdf/2306.03630.pdf)
## Set up

- pip install -r requirements.txt  
- cd ./first_stage/kernels/lib_tree_filter  
- python setup.py build develop  

## Train Model
- Prepare data for training the first stage (We provided the related data in:[train_data](https://drive.google.com/file/d/1up2LL6NpMMX38YeNa_6mWAayTyvkj9Mj/view?usp=sharing). Please download it and put it in the '/train_data/' folder, and download the VGG model and put it in './first_stage/vgg_models/': [vgg_pretrain_model](https://drive.google.com/file/d/1MnT9o84LRcp137eOA9JSljRJELe09TQ-/view?usp=sharing))  
- Run ./first_stage/train.py  
-  Prepare data for training the MVAE refine (We provided the pseodu label of train_data tested by the first_stage model in:[MVAE_refine](https://drive.google.com/file/d/1v678xKmDLzM6ZsKsH30rsKVXaS4swqj9/view?usp=sharing). Please download it and put it in the  "./first_stage/models/results_train/" folder)  
- Run ./MVAE_refine/train.py 
##  Test Model
- Run ./first_stage/test.py  
- Run ./MVAE_refine/test.py 

## Trained model:
Please download the trained model and put it in "./first_stage/models":  [[Google_Drive]](https://drive.google.com/file/d/1pGEclv5zNA878u2x5iCDcToC7hpmYNI1/view?usp=sharing);  
"./MVAE_refine/models": [[Google_Drive]](https://drive.google.com/file/d/1yTmmMu_ZsUrqkxb1zaR5mQXhHhfY4vP6/view?usp=sharing)
  
##  Prediction Maps
Results of our model on six benchmark datasets(NJU2K, SSB, DES, NLPR, LFSD, SIP) can be found: [first_stage](https://drive.google.com/file/d/12ZFJYII9j9_hCDVI307l6Bjd5YpDnMF2/view?usp=sharing);[MVAE_refine](https://drive.google.com/file/d/1gaJjDnUKktNhTbuGn7zJf3EFmlSrpBBF/view?usp=sharing)
 
