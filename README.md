# Transfer CLIP for Generalizable Image Denoising (CVPR2024)

> **Abstract:** *Image denoising is a fundamental task in computer vision. While prevailing deep learning-based supervised and self-supervised methods have excelled in eliminating in-distribution noise, their susceptibility to out-of-distribution (OOD) noise remains a significant challenge. The recent emergence of contrastive language-image pre-training (CLIP) model has showcased exceptional capabilities in open-world image recognition and segmentation. Yet, the potential for leveraging CLIP to enhance the robustness of low-level tasks remains largely unexplored. This paper uncovers that certain dense features extracted from the frozen ResNet image encoder of CLIP exhibit distortion-invariant and content-related properties, which are highly desirable for generalizable denoising. Leveraging these properties, we devise an asymmetrical encoder-decoder denoising network, which incorporates dense features including the noisy image and its multi-scale features from the frozen ResNet encoder of CLIP into a learnable image decoder to achieve generalizable denoising. The progressive feature augmentation strategy is further proposed to mitigate feature overfitting and improve the robustness of the learnable decoder. Extensive experiments and comparisons conducted across diverse OOD noises, including synthetic noise, real-world sRGB noise, and low-dose CT image noise, demonstrate the superior generalization ability of our method.* 
<hr />


## Data preparation

- For synthetic noise removal, we used clean images from CBSD432 and added i.i.d. Gaussian noise with sigma=15
- For sRGB noise removal, we utilized the ISP/inverse ISP pipelines from CBDNet and clean images from DIV2K, and synthesized noisy images based on Poisson-Gaussian noise with the fixed noise level. We use the following code:
```
python basicsr/data/ISP_implement.py
```
- For LDCT image noise removal, we used 1mm-thickness normal-dose CT images from nine patients (except for patient L506, which was used for test) of [Mayo2016-AAPM dataset](https://aapm.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h) and added i.i.d. Gaussian noise with sigma=5.

## Training and evaluation

Training and test commands for synthetic noise removal:
```
python basicsr/train.py -opt Denoising/Options/CLIPDenoising_SyntheticDenoising_GaussianSigma15.yml
python Denoising/test_synthetic_denoising.py
```

Training and test commands for sRGB noise removal:
```
python basicsr/train.py -opt Denoising/Options/CLIPDenoising_sRGBDenoising_FixedPoissonGaussian.yml
python Denoising/test_real_denoising_sRGB.py
```

Training and test  commands for LDCT image noise removal:
```
python basicsr/train.py -opt Denoising/Options/CLIPDenoising_LDCTDenoising_GaussianSigma5.yml
python Denoising/test_real_denoising_CT.py
```

### Pre-trained models
Pre-trained models can be found in https://drive.google.com/drive/folders/1F8md2zen0iPlhGnI5-VdjdBOLY8DBIiF?usp=drive_link

## Citation
Please consider citing this paper if it helps you:

    @inproceedings{Jun2024Transfer,
        title={Transfer CLIP for Generalizable Image Denoising}, 
        author={Jun Cheng and Dong Liang and Shan Tan},
        booktitle={CVPR},
        year={2024}
    }



**Acknowledgment:** This code is based on the [Restormer](https://github.com/swz30/Restormer) 
