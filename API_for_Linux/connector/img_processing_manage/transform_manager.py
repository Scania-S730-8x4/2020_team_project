from image_2_style_gan.align_images import align_images
import cv2
import os

def transform_jpg_to_png(jpg_path, png_path):
    try:
        cnv_buffer = cv2.imread(jpg_path)
        cv2.imwrite(png_path, cnv_buffer)
        os.remove(jpg_path)
        return png_path
    except Exception as e:
        print(f"origin image transform or aligned failed {e}")


def transform_aligned_custom_img(CUSTOM_DIR):
    try:
        cnv_buffer = cv2.imread(f'{CUSTOM_DIR}c_target.jpg')
        cv2.imwrite(f'{CUSTOM_DIR}un_aligned/c_target.png', cnv_buffer)
        os.remove(f'{CUSTOM_DIR}c_target.jpg')

        align_images(f'{CUSTOM_DIR}un_aligned/', f'{CUSTOM_DIR}aligned/')
    except Exception as e:
        print(f"custom image transform or aligned failed {e}")