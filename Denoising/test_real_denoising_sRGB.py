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
checkpoint = torch.load('./Denoising/pretrained_models/sRGB/net_g_latest.pth')
load_result = model_restoration.load_state_dict(checkpoint['params'])

model_restoration.cuda()
model_restoration.eval()
##########################

factor = 32

testsets = ['CC', 'PolyU', 'SIDD_val']

for testset in testsets:

    if testset == 'CC':
        noises = sorted(glob(os.path.join(args.input_dir, testset, '*real.png') ))
        cleans = sorted(glob(os.path.join(args.input_dir, testset, '*mean.png') ))
        index = -9
    
    elif testset == 'PolyU':
        noises = sorted(glob(os.path.join(args.input_dir, testset, '*real.JPG') ))
        cleans = sorted(glob(os.path.join(args.input_dir, testset, '*mean.JPG') ))
        index = -9

    elif testset == 'SIDD_val':
        cleans = sorted(glob(os.path.join(args.input_dir, 'SSID/SIDD_test_PNG', 'GT/*.png')))
        noises = sorted(glob(os.path.join(args.input_dir, 'SSID/SIDD_test_PNG', 'noisy/*.png')))

    psnr_list = []; ssim_list = []
    for noise, clean in tqdm(zip(noises, cleans)):
        
        with torch.no_grad():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            img_clean = utils.load_img(clean)

            img = np.float32(utils.load_img(noise))/255.

            img = torch.from_numpy(img).permute(2,0,1)
            input_ = img.unsqueeze(0).cuda()

            # Padding in case images are not multiples of 8
            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

            restored = model_restoration(input_)

            # Unpad images to original dimensions
            restored = restored[:,:,:h,:w]

            restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            restored = (restored * 255.0).round().astype(np.uint8)
            
            psnr, ssim = proc(img_clean, restored)

            psnr_list.append(psnr); ssim_list.append(ssim)

    print('dataset:{}, psnr:{:.2f}, ssim:{:.3f}'.format(testset, 
                            sum(psnr_list)/len(psnr_list), sum(ssim_list)/len(ssim_list)))
