import argparse
import numpy as np
import os
import sys

sys.path.append('/home/cj/code/CLIPDenoising')
import torch
import torch.nn.functional as F
import utils
from glob import glob
from scipy.ndimage import convolve
from tqdm import tqdm

from basicsr.models.archs.CLIPDenoising_arch import CLIPDenoising
from basicsr.metrics.CT_psnr_ssim import compute_PSNR, compute_SSIM

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Synthetic Color Denoising')

parser.add_argument('--input_dir', default='/data0/cj/dataset/AAPM_full/processed_image_data/processed_Image_data/1mm', type=str, help='Directory of validation images')

args = parser.parse_args()

def proc(tar_img, prd_img):        
    PSNR = utils.calculate_psnr(tar_img, prd_img)
    SSIM = utils.calculate_ssim(tar_img, prd_img)
    return PSNR, SSIM

# network arch
'''
type: CLIPDenoising
inp_channels: 1
out_channels: 1
depth: 5
wf: 64 
num_blocks: [3, 4, 6, 3] 
bias: false
model_path: /data0/cj/model_data/ldm/stable-diffusion/RN50.pt

aug_level: 0.025
'''

model_restoration = CLIPDenoising(inp_channels=1, out_channels=1, depth=5, wf=64, num_blocks=[3,4,6,3], bias=False,
                                  model_path='/data0/cj/model_data/ldm/stable-diffusion/RN50.pt', aug_level=0.025)
checkpoint = torch.load('./Denoising/pretrained_models/LDCT/net_g_latest.pth')
load_result = model_restoration.load_state_dict(checkpoint['params'])

model_restoration.cuda()
model_restoration.eval()
##########################

factor = 32

test_patient = 'L506'
target_path = sorted(glob(os.path.join(args.input_dir, '*target*')))
input_path = sorted(glob(os.path.join(args.input_dir, '*input*')))

input_ = [f for f in input_path if test_patient in f]
target_ = [f for f in target_path if test_patient in f]


trunc_min = -1024; trunc_max = 3072; shape_ = 512
psnr_list = []; ssim_list = []

for noise, clean in tqdm(zip(input_, target_)):
    
    with torch.no_grad():
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img_clean = np.load(clean)[..., np.newaxis]
        img_clean = torch.from_numpy(img_clean).permute(2,0,1)
        img_clean = img_clean.unsqueeze(0).float().cuda()

        img = np.load(noise)[..., np.newaxis]
        img = torch.from_numpy(img).permute(2,0,1)
        input_ = img.unsqueeze(0).float().cuda()

        # Padding in case images are not multiples of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        restored = model_restoration(input_)

        # Unpad images to original dimensions
        restored = restored[:,:,:h,:w]
        
        psnr, ssim = compute_PSNR(restored, img_clean, data_range=trunc_max-trunc_min, trunc_min=trunc_min, trunc_max=trunc_max,
                                 norm_range_max=3096, norm_range_min=-1024), \
                     compute_SSIM(restored, img_clean, data_range=trunc_max-trunc_min, trunc_min=trunc_min, trunc_max=trunc_max,
                                 norm_range_max=3096, norm_range_min=-1024)

        psnr_list.append(psnr); ssim_list.append(ssim)
        

print('CT dataset: psnr:{:.2f}, ssim:{:.3f}'.format(
                        sum(psnr_list)/len(psnr_list), sum(ssim_list)/len(ssim_list)))
