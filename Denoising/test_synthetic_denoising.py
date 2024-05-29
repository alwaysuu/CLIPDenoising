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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='Synthetic Color Denoising')

parser.add_argument('--input_dir', default='/data0/cj/dataset', type=str, help='Directory of validation images')

args = parser.parse_args()

def proc(tar_img, prd_img):        
    PSNR = utils.calculate_psnr(tar_img, prd_img)
    SSIM = utils.calculate_ssim(tar_img, prd_img)
    return PSNR, SSIM

# network arch
'''
type: CLIPDenoising
inp_channels: 3
out_channels: 3
depth: 5
wf: 64 
num_blocks: [3, 4, 6, 3] 
bias: false
model_path: /data0/cj/model_data/ldm/stable-diffusion/RN50.pt

aug_level: 0.025
'''

model_restoration = CLIPDenoising(inp_channels=3, out_channels=3, depth=5, wf=64, num_blocks=[3,4,6,3], bias=False,
                                  model_path='/data0/cj/model_data/ldm/stable-diffusion/RN50.pt', aug_level=0.025)
checkpoint = torch.load('./Denoising/pretrained_models/synthetic/net_g_latest.pth')
load_result = model_restoration.load_state_dict(checkpoint['params'])

model_restoration.cuda()
model_restoration.eval()
##########################

factor = 32

datasets = ['CBSD68', 'McM', 'Kodak', 'Urban100'] 
noise_types = ['gauss', 'spatial_gauss', 'poisson'] 

for dataset in datasets:
    for noise_type in noise_types:

        if noise_type == 'gauss':
            sigmas = [15, 25, 50]
        elif noise_type == 'poisson':
            sigmas = [2, 2.5, 3, 3.5]
        elif noise_type == 'spatial_gauss':
            sigmas = [40, 45, 50, 55]

        
        for sigma_test in sigmas:
            psnr_list = []; ssim_list = []
            inp_dir = os.path.join(args.input_dir, dataset)
            files = glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.tif'))

            with torch.no_grad():
                for file_ in tqdm(files):
                    img_clean = np.float32(utils.load_img(file_))/255.

                    np.random.seed(seed=0)  # for reproducibility

                    # gaussian noise
                    if noise_type == 'gauss':
                        img = img_clean + np.random.normal(0, sigma_test/255., img_clean.shape)

                    # poisson noise
                    elif noise_type == 'poisson':
                        img = utils.add_poisson_noise(img_clean, scale=sigma_test)

                    elif noise_type == 'spatial_gauss':
                        noise = np.random.normal(0, sigma_test/255., img_clean.shape)
                        kernel = np.ones((3,3))/9.0
                        for chn in range(img_clean.shape[-1]):
                            noise[...,chn] = convolve(noise[...,chn], kernel)

                        img = img_clean + noise

                    img = torch.from_numpy(img).permute(2,0,1).float()
                    input_ = img.unsqueeze(0).cuda()

                    # Padding in case images are not multiples of 8
                    h,w = input_.shape[2], input_.shape[3]
                    H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
                    padh = H-h if h%factor!=0 else 0
                    padw = W-w if w%factor!=0 else 0
                    input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

                    restored = model_restoration(input_)
                    restored = restored[:,:,:h,:w]

                    restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                    psnr, ssim = proc(img_clean*255.0, restored*255.0)

                    psnr_list.append(psnr); ssim_list.append(ssim)

            print('noise_type:{}, dataset:{}, sigma:{}, psnr:{:.2f}, ssim:{:.3f}'.format(noise_type, dataset, sigma_test, 
                                    sum(psnr_list)/len(psnr_list), sum(ssim_list)/len(ssim_list)))
