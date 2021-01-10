from image_2_style_gan.read_image import image_reader_color
from image_2_style_gan.align_images import align_images

from skimage.exposure import match_histograms
from torchvision.utils import save_image
from skimage.io import imread, imsave
import numpy as np
import argparse
import torch
import os

ch_color_mean = []

def custom_channel_manipulator(src_dir, trg_dir):
    raw_img_file_name = os.listdir(src_dir)[0]
    raw_image = image_reader_color(src_dir + raw_img_file_name)

    for i in range(3):
        ch_color_mean.append(torch.mean(raw_image[:, :, i]))

    idx_min = ch_color_mean.index(min(ch_color_mean))
    idx_max = ch_color_mean.index(max(ch_color_mean))
    bias = (max(ch_color_mean) - min(ch_color_mean))

    print(min(ch_color_mean))
    print(max(ch_color_mean))
    print(bias)
    raw_image -= bias * 10

    save_image(torch.clamp(raw_image, 0, 1), trg_dir + '00.png')

    return raw_image


def target_channel_manipulator(img_trg, img_org, blur_mask):
    hist_bias = torch.Tensor([0, 0, 0])
    for i in range(3):
        hist_bias[i] = torch.mean(img_org[0, i, ..., ...]*blur_mask) - torch.mean(img_trg[0, i, ..., ...]*blur_mask)
        multiplier = 1
        if hist_bias[i] < 0:
            if hist_bias[i] >= -0.001:
                img_trg += hist_bias[i]*0
            elif hist_bias[i] >= -0.002 and hist_bias[i] < -0.001:
                img_trg += hist_bias[i]
            elif hist_bias[i] >= -0.003 and hist_bias[i] < -0.002:
                img_trg += hist_bias[i]*2
            else:
                img_trg += hist_bias[i]*5
            multiplier = -1
        img_trg[0, i, ..., ...] = torch.clamp(img_trg[0, i, ..., ...] + hist_bias[i]*multiplier, 0, 1)
    
    return img_trg


def color_histogram_matching(origin_name, target_name, save_file_name):
    compare_origin = imread(origin_name).astype(np.uint8)
    compare_target = imread(target_name).astype(np.uint8)
    matched = match_histograms(compare_origin, compare_target, multichannel=True).astype(np.uint8)
    imsave(save_file_name, matched)

    return matched


def eyes_channel_matching(img_org, blur_mask_eyes):
    ingredient_eyes = img_org * blur_mask_eyes
    blur_mask_invt = 1 - blur_mask_eyes
    e_mean = []
    for i in range(ingredient_eyes.shape[1]):
        e_mean.append(torch.mean(ingredient_eyes[0, i, ..., ...]))
    idx_min = e_mean.index(min(e_mean))
    idx_max = e_mean.index(max(e_mean))

    if min(e_mean) > 0.001:
        for i in range(ingredient_eyes.shape[1]):
            ingredient_eyes[0, i, ..., ...] = torch.clamp(
                ingredient_eyes[0, idx_min, ..., ...] + ingredient_eyes[0, idx_min, ..., ...]**(i+4), 0, 1)
    elif min(e_mean) > 0.0005 and min(e_mean) < 0.001:
        for i in range(ingredient_eyes.shape[1]):
            ingredient_eyes[0, i, ..., ...] = torch.clamp(
                ingredient_eyes[0, idx_min, ..., ...] + ingredient_eyes[0, idx_max, ..., ...]**(i+3), 0, 1)
    else:
        for i in range(ingredient_eyes.shape[1]):
            ingredient_eyes[0, i, ..., ...] = torch.clamp(
                ingredient_eyes[0, idx_min, ..., ...] + ingredient_eyes[0, idx_max, ..., ...]**(i+2), 0, 1)

    # for i in range(ingredient_eyes.shape[1]):
    #     ingredient_eyes[0, i, ..., ...] += torch.clamp(ingredient_eyes[0, idx_max, ..., ...]**((i+1)*multiplier), 0, 1)
    #         #ingredient_eyes[0, idx_min, ..., ...] + 
            
    img_org = (img_org * blur_mask_invt) + ingredient_eyes

    return img_org


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Source_dir, Target_dir')
    parser.add_argument('--src_dir', default='images/raw/')
    parser.add_argument('--cm_dir', default='images/ch_matched/')
    parser.add_argument('--trg_dir', default='images/aligned/')
    parser.add_argument('--crop', default=False)
    args = parser.parse_args()

    src_dir = args.src_dir
    cm_dir = args.cm_dir
    trg_dir = args.trg_dir
    crop_flag = args.crop

    if not os.path.isdir(src_dir):
            os.makedirs(src_dir, exist_ok=True)

    if not os.path.isdir(cm_dir):
            os.makedirs(cm_dir, exist_ok=True)

    if not os.path.isdir(trg_dir):
            os.makedirs(trg_dir, exist_ok=True)

    custom_channel_manipulator(src_dir, cm_dir)
    if crop_flag:
        align_images(cm_dir, trg_dir)