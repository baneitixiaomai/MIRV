## Mutual Information Regularization for Weakly-supervised RGB-D Salient Object Detection
## Set up
  pip install -r requirements.txt
  cd ./first_stage/kernels/lib_tree_filter
  python setup.py build develop
## Data preparation
1. Please download data and put it in  '/train_data/' [train_data](https://drive.google.com/file/d/1up2LL6NpMMX38YeNa_6mWAayTyvkj9Mj/view?usp=sharing)  
2. Please download the pseodu label of train_data tested by the first_stage model and put it in  "./first_stage/models/results_train/" [MVAE_refine](https://drive.google.com/file/d/1v678xKmDLzM6ZsKsH30rsKVXaS4swqj9/view?usp=sharing)  
3. Please download the VGG model and put it in './first_stage/vgg_models/': [vgg_pretrain_model](https://drive.google.com/file/d/1MnT9o84LRcp137eOA9JSljRJELe09TQ-/view?usp=sharing)  

## Trained model:
Please download the trained model and put it in "./first_stage/models": [first_stage](https://drive.google.com/file/d/1pGEclv5zNA878u2x5iCDcToC7hpmYNI1/view?usp=sharing);
"./MVAE_refine/models": [MVAE_refine](https://drive.google.com/file/d/1yTmmMu_ZsUrqkxb1zaR5mQXhHhfY4vP6/view?usp=sharing)
  
##  Prediction Maps
Results of our model on six benchmark datasets can be found: [first_stage](https://drive.google.com/file/d/12ZFJYII9j9_hCDVI307l6Bjd5YpDnMF2/view?usp=sharing);[MVAE_refine](https://drive.google.com/file/d/1gaJjDnUKktNhTbuGn7zJf3EFmlSrpBBF/view?usp=sharing)
 
