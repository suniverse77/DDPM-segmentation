# Diffusion-Based Semantic Segmentation Using Feature Regularization and Soft Voting

Summer Annual Conference of IEIE 2025

**Note:** use **--recurse-submodules** when clone.

&nbsp;
## Model Architecture

<center><img src='{{"/images/DDPM-segmentation1.png" | relative_url}}' width="100%"></center>

&nbsp;
## Datasets

The evaluation is performed on 6 collected datasets with a few annotated images in the training set:
Bedroom-18, FFHQ-34, Cat-15, Horse-21, CelebA-19 and ADE-Bedroom-30. The number corresponds to the number of semantic classes.

[datasets.tar.gz](https://storage.yandexcloud.net/yandex-research/ddpm-segmentation/datasets.tar.gz) (~47Mb)


&nbsp;
## DDPM

### Pretrained DDPMs

The models trained on LSUN are adopted from [guided-diffusion](https://github.com/openai/guided-diffusion).

### Run 

1. Download the datasets:\
 &nbsp;&nbsp;```bash datasets/download_datasets.sh```
2. Download the DDPM checkpoint:\
 &nbsp;&nbsp; ```bash checkpoints/ddpm/download_checkpoint.sh <checkpoint_name>```
3. Check paths in ```experiments/<dataset_name>/ddpm.json``` 
4. Run: ```bash scripts/ddpm/train_interpreter.sh <dataset_name>```
   
**Available checkpoint names:** lsun_bedroom, ffhq, lsun_cat, lsun_horse\
**Available dataset names:** bedroom_28, ffhq_34, cat_15, horse_21, celeba_19, ade_bedroom_30

**Note:** ```train_interpreter.sh``` is RAM consuming since it keeps all training pixel representations in memory. For ex, it requires ~210Gb for 50 training images of 256x256. (See [issue](https://github.com/nv-tlabs/datasetGAN_release/issues/34))

**Pretrained pixel classifiers** and test predictions are [here](https://www.dropbox.com/s/kap229jvmhfwh7i/pixel_classifiers.tar?dl=0).



&nbsp;
## Results

Performance in terms of mean IoU:

| Method       | DDPM   | DDPM_V    | DDPM_E | **DDPM_VE**
|:------------ |:------ |:--------- |:------ |:--------- |
| Bedroom-28   | 45.45  | 48.24  	| 47.40 	| **50.09** |
| FFHQ-34 	   | 56.53  | **58.22** | 56.76  | 58.07  	|
| Cat-15       | 55.38 	| 58.12   	| 56.51 	| **59.28** |
| Horse-21     | 60.83 	| 63.93   	| 61.69 	| **64.21** |
| CelebA-19    | 56.70 	| 58.31   	| 56.92	| **58.51** | 

- DDPM_V is a model using soft voting.
- DDPM_E is a model using ETF Regularizer.
- DDPM_VE is a model using both ETF Regularizer and soft voting.

&nbsp;
Examples of segmentation masks predicted by the DDPM-based method:

<center><img src='{{"/images/DDPM-segmentation2.png" | relative_url}}' width="100%"></center>
