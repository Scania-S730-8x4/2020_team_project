import random
import os
import shutil
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from image_2_style_gan.align_images import align_images
from image_2_style_gan.mask_makers.precision_facial_mask_maker import precision_facial_masks
from image_2_style_gan.color_channel_manipulator import *
from image_2_style_gan.random_noise_image import random_pixel_image
from image_2_style_gan.read_image import *
from image_2_style_gan.perceptual_model import VGG16_for_Perceptual
from image_2_style_gan.stylegan_layers import G_mapping, G_synthesis
from torchvision.utils import save_image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def image_crossover_eyes(BASE_DIR, RAW_DIR, rand_uuid, process_selection, gender):
    ALIGNED_IMAGE_DIR = f'{BASE_DIR}aligned/'
    os.mkdir(ALIGNED_IMAGE_DIR)

    TARGET_IMAGE_DIR = f'{BASE_DIR}target/'
    os.mkdir(TARGET_IMAGE_DIR)

    ITERATION = 150
    BATCH_SIZE = 1  # Not in usage.

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

    model_resolution = 1024

    aligned_image_names = align_images(RAW_DIR, ALIGNED_IMAGE_DIR)

    try:
        if not aligned_image_names:
            print('\nFailed Face Alignment. Abort process.')
            shutil.rmtree(BASE_DIR)
            sys.exit(1)

        aligned_image_name = ALIGNED_IMAGE_DIR + \
            os.listdir(ALIGNED_IMAGE_DIR)[0]
        mask_name, face_mask_name, eyes_mask_name, brows_mask_name, lids_mask_name = precision_facial_masks(
            aligned_image_name, MASK_DIR)

        ingredient_name = ALIGNED_IMAGE_DIR + os.listdir(ALIGNED_IMAGE_DIR)[0]
        random_target_image_index = random.randint(
            0, len(os.listdir(TARGET_SOURCE_DIR))-1)
        target_name = TARGET_SOURCE_DIR + \
            os.listdir(TARGET_SOURCE_DIR)[random_target_image_index]

    except IndexError as e:
        print("Missing file(s).\nCheck if all of source images prepared properly and try again.")
        print(f"Error of [ Alignment and Mask Creation ] part : {e}")
        shutil.rmtree(BASE_DIR)
        sys.exit(1)

    final_name = FINAL_IMAGE_DIR + str(rand_uuid) + '.png'

    g_all = nn.Sequential(OrderedDict([('g_mapping', G_mapping()),  # ('truncation', Truncation(avg_latent)),
                                       ('g_synthesis', G_synthesis(resolution=model_resolution))]))

    g_all.load_state_dict(torch.load(
        f"../image_2_style_gan/torch_weight_files/karras2019stylegan-ffhq-{model_resolution}x{model_resolution}.pt", map_location=device))
    g_all.eval()
    g_all.to(device)
    g_mapping, g_synthesis = g_all[0], g_all[1]

    img_0 = image_reader_color(target_name)  # (1,3,1024,1024) -1~1
    img_0 = img_0.to(device)

    img_1 = image_reader_color(ingredient_name)
    img_1 = img_1.to(device)  # (1,3,1024,1024)

    blur_mask0_1 = image_reader_gray(mask_name).to(device)
    blur_mask0_2 = image_reader_gray(eyes_mask_name).to(device)
    blur_mask0_3 = image_reader_gray(lids_mask_name).to(device)
    # blur_mask0_4 = image_reader_gray(face_mask_name).to(device)
    # blur_mask0_5 = image_reader_gray(brows_mask_name).to(device)
    
    blur_mask1 = 1-blur_mask0_1
    blur_mask_eyes = blur_mask0_1-blur_mask0_2
    blur_mask_lids = blur_mask0_1-torch.clamp(blur_mask0_3-blur_mask0_2, 0, 1)

    img_0 = target_channel_manipulator(img_0, img_1, blur_mask0_1)    
    img_1 = eyes_channel_matching(img_1, blur_mask0_2)
    save_image(img_0, '../image_2_style_gan/source/trghistadjs.png')
    save_image(img_1, '../image_2_style_gan/source/orghistadjs.png')
    save_image(blur_mask0_1, '../image_2_style_gan/source/mask.png')

    MSE_Loss = nn.MSELoss(reduction="mean")
    upsample2d = torch.nn.Upsample(scale_factor=0.5, mode='nearest')

    img_p0 = img_0.clone()  # resize for perceptual net
    img_p0 = upsample2d(img_p0)
    img_p0 = upsample2d(img_p0)  # (1,3,256,256)

    img_p1 = img_1.clone()
    img_p1 = upsample2d(img_p1)
    img_p1 = upsample2d(img_p1)  # (1,3,256,256)

    # img_noise = random_pixel_image(min_float=0.3, max_float=0.7).to(device)

    perceptual_net = VGG16_for_Perceptual(n_layers=[2, 4, 14, 21]).to(device)  # conv1_1,conv1_2,conv2_2,conv3_3
    dlatent = torch.zeros((1, 18, 512), requires_grad=True, device=device)  # requires_grad를 'True'로 두어 오차 역전파 과정 중 해당 Tensor에 대한 변화도를 계산하도록 한다. 
    optimizer = optim.Adam({dlatent}, lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    loss_list = []

    print("Start ---------------------------------------------------------------------------------------------")
    # [img_0 : Target IMG] / [img_1 : Ingredient IMG]
    for i in range(ITERATION):
        img_noise = random_pixel_image(min_float=0.3, max_float=0.7).to(device)
        
        optimizer.zero_grad()  # 매 Iter 시마다 가중치들의 변화도가 누적되지 않도록 그 변화도를 초기화시켜 준다.
        synth_img = g_synthesis(dlatent)
        synth_img = (synth_img*0.75 + img_noise*0.25) / 2
        
        # 랜덤 노이즈인 이미지로 시작하고, 이미 특정 DATA Set으로 학습된 모델을 사용한다.
        # 그렇기 때문에 오차 비교 부분을 없앨 경우, 내장된 가중치를 가지고 특정한 한 인물의 이미지만을 계속 만들어낸다.
        
        # 순전파 과정을 수행한다.
        loss_wl0 = caluclate_loss(synth_img, img_0, perceptual_net, img_p0, blur_mask_eyes, MSE_Loss, upsample2d)
        # 'loss_wl0'은 변화해 갈 synth_imge와 img_0(Target Image)에 각각 'blur_mask_eyes'마스크를 적용한 Image들 간의 오차를 계산해 낸다.
        loss_wl1 = caluclate_loss(synth_img, img_1, perceptual_net, img_p1, blur_mask_lids, MSE_Loss, upsample2d)
        # 'loss_wl1'은 변화해 갈 synth_imge와 img_1(Ingredient Image)에 각각 'blur_mask_lids'마스크를 적용한 Image들 간의 오차를 계산해 낸다.
        loss = loss_wl0 + loss_wl1
        # 최종 loss는 loss_wl0, loss_wl1 두 loss 모두를 더한 값이다.
        loss.backward()
        # 오차 역전파 과정을 진행한다.

        optimizer.step()
        # 가중치의 변화도를 .step()함수를 호출해 갱신한다.
        
        loss_np = loss.detach().cpu().numpy()
        loss_0 = loss_wl0.detach().cpu().numpy()
        loss_1 = loss_wl1.detach().cpu().numpy()
        # 위 순전파 과정에서 구해진 해당 Iter의 Loss값들을 연산 기록으로부터 분리한 후 별도의 변수에 따로 저장한다.

        loss_list.append(loss_np)

        if i % 10 == 0:
            print("iter{}: loss --{},  loss0 --{},  loss1 --{}".format(i, loss_np, loss_0, loss_1))
        elif i == (ITERATION - 1):
            save_image(img_1*blur_mask1 + synth_img*blur_mask0_1, final_name)

    origin_name = '{}{}_origin.png'.format(FINAL_IMAGE_DIR, str(rand_uuid))
    os.replace(ingredient_name, origin_name)

    print("Complete -------------------------------------------------------------------------------------")

    return origin_name, final_name


def caluclate_loss(synth_img, img, perceptual_net, img_p, blur_mask, MSE_Loss, upsample2d):  # W_l
    # calculate MSE Loss
    mse_loss = MSE_Loss(synth_img*blur_mask.expand(1, 3, 1024, 1024), img*blur_mask.expand(1, 3, 1024, 1024))  # (lamda_mse/N)*||G(w)-I||^2
                        
    # calculate Perceptual Loss
    real_0, real_1, real_2, real_3 = perceptual_net(img_p)
    synth_p = upsample2d(synth_img)  # (1,3,256,256)
    synth_p = upsample2d(synth_p)
    synth_0, synth_1, synth_2, synth_3 = perceptual_net(synth_p)

    perceptual_loss = 0
    blur_mask = upsample2d(blur_mask)
    blur_mask = upsample2d(blur_mask)  # (256,256)

    perceptual_loss += MSE_Loss(synth_0*blur_mask.expand(1, 64, 256, 256), real_0*blur_mask.expand(1, 64, 256, 256))
    perceptual_loss += MSE_Loss(synth_1*blur_mask.expand(1, 64, 256, 256), real_1*blur_mask.expand(1, 64, 256, 256))

    blur_mask = upsample2d(blur_mask)
    blur_mask = upsample2d(blur_mask)  # (64,64)
    perceptual_loss += MSE_Loss(synth_2*blur_mask.expand(1, 256, 64, 64), real_2*blur_mask.expand(1, 256, 64, 64))

    blur_mask = upsample2d(blur_mask)  # (64,64)
    perceptual_loss += MSE_Loss(synth_3*blur_mask.expand(1, 512, 32, 32), real_3*blur_mask.expand(1, 512, 32, 32))

    return mse_loss+perceptual_loss
