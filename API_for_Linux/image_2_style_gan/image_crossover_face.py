import random
import os
import shutil
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from image_2_style_gan.align_images import align_images
from image_2_style_gan.mask_makers.mask_maker import mask_maker
from image_2_style_gan.read_image import image_reader_color
from image_2_style_gan.random_noise_image import random_pixel_image
from image_2_style_gan.perceptual_model import VGG16_for_Perceptual
from image_2_style_gan.stylegan_layers import G_mapping, G_synthesis
from torchvision.utils import save_image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def image_crossover_face(BASE_DIR, RAW_DIR, rand_uuid, process_selection, gender):    
    ALIGNED_IMAGE_DIR = f'{BASE_DIR}aligned/'
    os.mkdir(ALIGNED_IMAGE_DIR)
    
    TARGET_IMAGE_DIR = f'{BASE_DIR}target/'
    os.mkdir(TARGET_IMAGE_DIR)
    
    if process_selection == 0 and gender == 'female':
        TARGET_SOURCE_DIR = '../image_2_style_gan/source/target/female/'
    elif process_selection == 0 and gender == 'male':
        TARGET_SOURCE_DIR = '../image_2_style_gan/source/target/male/'
    else:
        TARGET_SOURCE_DIR = f'{BASE_DIR}raw_target/aligned/'

    FINAL_IMAGE_DIR = f'{BASE_DIR}final/'
    os.mkdir(FINAL_IMAGE_DIR)

    MASK_DIR = f'{BASE_DIR}mask/'
    os.mkdir(MASK_DIR)

    model_resolution=1024
    ITERATION = 150

    aligned_image_names = align_images(RAW_DIR, ALIGNED_IMAGE_DIR)

    try:
        if not aligned_image_names:
            print('\nFailed Face Alignment. Abort process.')
            shutil.rmtree(BASE_DIR) # UUID 디렉터리 삭제
            sys.exit(1)

        aligned_image_name = ALIGNED_IMAGE_DIR + os.listdir(ALIGNED_IMAGE_DIR)[0]

        mask_maker(aligned_image_name, MASK_DIR)

        ingredient_name = ALIGNED_IMAGE_DIR + os.listdir(ALIGNED_IMAGE_DIR)[0]

        random_target_image_index = random.randint(0, len(os.listdir(TARGET_SOURCE_DIR))-1)

        target_name = TARGET_SOURCE_DIR + os.listdir(TARGET_SOURCE_DIR)[random_target_image_index]
        target_number = os.listdir(TARGET_SOURCE_DIR)[random_target_image_index]
        print(f"target_number: {target_number}")

    except IndexError as e:
        print("\nMissing file(s).\nCheck if all of source images prepared properly and try again.")
        print(f"Aligned_image_names function: {e}")
        shutil.rmtree(BASE_DIR) # UUID 디렉터리 삭제
        sys.exit(1)

    try:
        mask_name = MASK_DIR + os.listdir(MASK_DIR)[0]
    except Exception as e:
        shutil.copyfile('../image_2_style_gan/source/ref_mask/ref_mask.png', '{}ref_mask.png'.format(MASK_DIR))
        mask_name = MASK_DIR + os.listdir(MASK_DIR)[0]

    final_name = FINAL_IMAGE_DIR + str(rand_uuid) + '.png'

    g_all = nn.Sequential(OrderedDict([('g_mapping', G_mapping()),# ('truncation', Truncation(avg_latent)),
    ('g_synthesis', G_synthesis(resolution=model_resolution))]))

    g_all.load_state_dict(torch.load( f"../image_2_style_gan/torch_weight_files/karras2019stylegan-ffhq-{model_resolution}x{model_resolution}.pt", map_location=device))
    g_all.eval()
    g_all.to(device)
    g_mapping,g_synthesis=g_all[0],g_all[1]

    img_0=image_reader_color(target_name) #(1,3,1024,1024) -1~1
    img_0=img_0.to(device)

    img_1=image_reader_color(ingredient_name)
    img_1=img_1.to(device) #(1,3,1024,1024)

    save_image(img_0, '../image_2_style_gan/source/trghistadjs.png')
    save_image(img_1, '../image_2_style_gan/source/orghistadjs.png')

    blur_mask0=image_reader_color(mask_name).to(device)
    blur_mask0=blur_mask0[:,0,:,:].unsqueeze(0)
    blur_mask1=blur_mask0.clone()
    blur_mask1=1-blur_mask0

    MSE_Loss=nn.MSELoss(reduction="mean")
    upsample2d=torch.nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)

    img_p0=img_0.clone() #resize for perceptual net
    img_p0=upsample2d(img_p0)
    img_p0=upsample2d(img_p0) #(1,3,256,256)

    img_p1=img_1.clone()
    img_p1=upsample2d(img_p1)
    img_p1=upsample2d(img_p1) #(1,3,256,256)

    perceptual_net=VGG16_for_Perceptual(n_layers=[2,4,14,21]).to(device) #conv1_1,conv1_2,conv2_2,conv3_3
    dlatent=torch.zeros((1,18,512), requires_grad=True, device=device)
    optimizer=optim.Adam({dlatent}, lr=0.01, betas=(0.9,0.999), eps=1e-8)

    loss_list = []

    print("Start ---------------------------------------------------------------------------------------------")
    for i in range(ITERATION):  # [img_0 : Target IMG] / [img_1 : Ingredient IMG]
        img_noise = random_pixel_image(min_float=0.2, max_float=0.8).to(device)
        optimizer.zero_grad()
        synth_img=g_synthesis(dlatent)
        synth_img=(synth_img*0.8 + img_noise*0.2) / 2

        loss_wl0=caluclate_loss(synth_img,img_0,perceptual_net,img_p0,blur_mask0,MSE_Loss,upsample2d)
        loss_wl1=caluclate_loss(synth_img,img_1,perceptual_net,img_p1,blur_mask1,MSE_Loss,upsample2d)
        loss=loss_wl0 + loss_wl1

        loss.backward()

        optimizer.step()

        loss_np=loss.detach().cpu().numpy()
        loss_0=loss_wl0.detach().cpu().numpy()
        loss_1=loss_wl1.detach().cpu().numpy()

        loss_list.append(loss_np)

        if i % 10 == 0:
            print("iter{}: loss --{},  loss0 --{},  loss1 --{}".format(i, loss_np, loss_0, loss_1))
        elif i == (ITERATION - 1):
            save_image(synth_img.clamp(0, 1), final_name)

    origin_name = '{}{}_origin.png'.format(FINAL_IMAGE_DIR, str(rand_uuid))
    os.replace(ingredient_name, origin_name)
    os.remove(mask_name) # 마스크 파일 삭제

    print("Complete -------------------------------------------------------------------------------------")

    return origin_name, final_name


def caluclate_loss(synth_img,img,perceptual_net,img_p,blur_mask,MSE_Loss,upsample2d): #W_l
    #calculate MSE Loss
    mse_loss=MSE_Loss(synth_img*blur_mask.expand(1,3,1024,1024),img*blur_mask.expand(1,3,1024,1024)) # (lamda_mse/N)*||G(w)-I||^2
    
    #calculate Perceptual Loss
    real_0,real_1,real_2,real_3=perceptual_net(img_p)
    synth_p=upsample2d(synth_img) #(1,3,256,256)
    synth_p=upsample2d(synth_p)
    synth_0,synth_1,synth_2,synth_3=perceptual_net(synth_p)

    perceptual_loss=0
    blur_mask=upsample2d(blur_mask)
    blur_mask=upsample2d(blur_mask) #(256,256)
    perceptual_loss+=MSE_Loss(synth_0*blur_mask.expand(1,64,256,256),real_0*blur_mask.expand(1,64,256,256))
    perceptual_loss+=MSE_Loss(synth_1*blur_mask.expand(1,64,256,256),real_1*blur_mask.expand(1,64,256,256))

    blur_mask=upsample2d(blur_mask)
    blur_mask=upsample2d(blur_mask) #(64,64)
    perceptual_loss+=MSE_Loss(synth_2*blur_mask.expand(1,256,64,64),real_2*blur_mask.expand(1,256,64,64))

    blur_mask=upsample2d(blur_mask) #(64,64)
    perceptual_loss+=MSE_Loss(synth_3*blur_mask.expand(1,512,32,32),real_3*blur_mask.expand(1,512,32,32))

    return mse_loss+perceptual_loss


if __name__ == "__main__":
    image_crossover_face()