import shutil
from ..img_processing_manage.transform_manager import *
from ..img_processing_manage.save_photo import save_jpg
import os


def make_dir(process_selection, rand_uuid):
    BASE_DIR = f'../image_2_style_gan/images/{rand_uuid}/'
    if process_selection == 0:
        if os.path.isdir(BASE_DIR) is not True:
            os.makedirs(BASE_DIR, exist_ok=True)

        RAW_DIR = f'{BASE_DIR}raw/'
        os.mkdir(RAW_DIR)
        return BASE_DIR, RAW_DIR

    if process_selection == 1:
        CUSTOM_DIR = f'{BASE_DIR}raw_target/'
        os.makedirs(f'{CUSTOM_DIR}aligned/', exist_ok=True)
        os.mkdir(f'{CUSTOM_DIR}un_aligned/')
        return BASE_DIR, CUSTOM_DIR


def make_img_path(RAW_DIR, rand_uuid):
    jpg_path = f'{RAW_DIR}raw_{rand_uuid}.jpg'
    png_path = f'{RAW_DIR}{rand_uuid}.png'
    return jpg_path, png_path


def origin_image_control(data, process_selection, rand_uuid):
    try:
        # 베이스 디렉터리 생성
        BASE_DIR, RAW_DIR = make_dir(process_selection, rand_uuid)
        # 경로 설정
        jpg_path, png_path = make_img_path(RAW_DIR, rand_uuid)
        # jpg 저장
        save_jpg(jpg_path, data['origin'], process_selection)
        # jpg -> png
        png_path = transform_jpg_to_png(jpg_path, png_path)
        return BASE_DIR, RAW_DIR

    except Exception as e:
        shutil.rmtree(BASE_DIR)
        print(f"origin processing failed : {e}")


def custom_image_control(data, process_selection, rand_uuid):
    try:
        # 디렉터리 생성
        _, CUSTOM_DIR = make_dir(process_selection, rand_uuid)
        # 커스텀 jpg 저장
        save_jpg(CUSTOM_DIR, data['custom'], process_selection)
        # jpg -> png, 정방형 처리
        transform_aligned_custom_img(CUSTOM_DIR)

    except Exception as e:
        shutil.rmtree(CUSTOM_DIR)
        print(f"custom processing failed : {e}")
