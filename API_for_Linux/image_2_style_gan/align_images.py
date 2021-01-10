import os
from image_2_style_gan.face_alignment import face_align
from image_2_style_gan.landmarks_detector import LandmarksDetector


def align_images(RAW_IMAGE_DIR, ALIGNED_IMAGE_DIR):
    landmarks_model_path = '../image_2_style_gan/landmark_model/shape_predictor_68_face_landmarks.dat'
    landmarks_detector = LandmarksDetector(landmarks_model_path)
    alinged_files = []

    for img_name in os.listdir(RAW_IMAGE_DIR):
        if img_name == '':
            return
        raw_img_path = os.path.join(RAW_IMAGE_DIR, img_name)
        for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
            aligned_face_path = '{}{}_a.png'.format(ALIGNED_IMAGE_DIR, os.path.splitext(img_name)[0])
            alinged_files.append(face_align(raw_img_path, aligned_face_path, face_landmarks))

    return alinged_files


if __name__ == "__main__":
    RAW_IMAGE_DIR = 'images/raw/'
    ALIGNED_IMAGE_DIR = 'images/aligned/'
    
    if not os.path.isdir(RAW_IMAGE_DIR):
            os.makedirs(RAW_IMAGE_DIR, exist_ok=True)

    if not os.path.isdir(ALIGNED_IMAGE_DIR):
            os.makedirs(ALIGNED_IMAGE_DIR, exist_ok=True)

    align_images(RAW_IMAGE_DIR, ALIGNED_IMAGE_DIR)
