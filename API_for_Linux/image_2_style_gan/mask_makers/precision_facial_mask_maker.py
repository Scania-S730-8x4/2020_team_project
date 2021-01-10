from collections import OrderedDict

import numpy as np
import argparse
import dlib
import cv2
import os

default_dir = '../image_2_style_gan/landmark_model/shape_predictor_68_face_landmarks.dat'


def precision_facial_masks(aligned_image_name, mask_dir, landmark_dat_dir=default_dir):
    FACIAL_LANDMARKS_INDEXES = OrderedDict([
    # ("Mouth", (48, 68)),
    ("Right_Eyebrow", (17, 22)),
    ("Left_Eyebrow", (22, 27)),
    ("Right_Eye", (36, 42)),
    ("Left_Eye", (42, 48))#,
    # ("Nose", (27, 35)),
    # ("Jaw", (0, 17))
    ])

    FACIAL_CONVPOLY_INDEXES = []
    FACIAL_CONVPOLY_INDEXES.append(range(17, 27))
    FACIAL_CONVPOLY_INDEXES.append(range(16, -1, -1))

    bias = [[-40, 0], [-30, -28], [20, -28], [25, 0], [0, 10], [0, 10], [-25, 0], [-20, -28], [30, -28], [40, 0], [0, 10], [0, 10]]
    eye_only_bias = [[1, 0], [0, 0], [0, 0], [0, -1], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [-1, 0], [0, 0], [0, 0]]
    lid_bias = [[-7, -15], [0, -13], [0, -13], [-3, -12], [3, -12], [0, -13], [0, -13], [7, -15]]
    brows_bias = [0, 20]

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(landmark_dat_dir)

    image = cv2.imread(aligned_image_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    shape = predictor(gray, detector(gray, 1)[0])
    coordinates = np.zeros((68, 2), dtype=int)

    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    mask_base = np.zeros((1024, 1024, 1))
    face_base = np.zeros((1024, 1024, 1))
    eyes_base = np.zeros((1024, 1024, 1))
    lids_base = np.zeros((1024, 1024, 1))
    brows_base = np.zeros((1024, 1024, 1))

    for i, landmark_name in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
        (j, k) = FACIAL_LANDMARKS_INDEXES[landmark_name]
        if landmark_name == 'Right_Eyebrow' or landmark_name == 'Left_Eyebrow':
            brows_coords = np.append(coordinates[j:k], np.flip(coordinates[j:k] + brows_bias, axis=0), axis=0)
            brows_base = cv2.fillConvexPoly(brows_base, brows_coords, 255)
        else:
            lid_coords = np.append(coordinates[j: k-2], np.flip(coordinates[j: k-2] + lid_bias[0+(4*(i-2)):4+(4*(i-2))], axis=0), axis=0)
            lids_base = cv2.fillConvexPoly(lids_base, lid_coords, 255)
            mask_base = cv2.fillConvexPoly(mask_base, coordinates[j:k] + bias[0+(6*(i-2)):6+(6*(i-2))], 255)
            eyes_base = cv2.fillConvexPoly(eyes_base, coordinates[j:k] + eye_only_bias[0+(6*(i-2)):6+(6*(i-2))], 255)
    
    face_poly_coords = []
    for i in FACIAL_CONVPOLY_INDEXES:
        for x, y in coordinates[i]:
            face_poly_coords.insert(len(face_poly_coords), [x, y])
    face_base = cv2.fillConvexPoly(face_base, np.array(face_poly_coords), 255)

    face_base = cv2.blur(face_base, (30, 30))
    mask_base = cv2.blur(mask_base, (15,20))
    eyes_base = cv2.blur(eyes_base, (2,2))
    brows_base = cv2.blur(brows_base, (10,15))
    lids_base = cv2.blur(lids_base, (5,7))

    cv2.imwrite(mask_dir + 'mask.png', mask_base)
    cv2.imwrite(mask_dir + 'mask_face.png', face_base)
    cv2.imwrite(mask_dir + 'mask_eyes.png', eyes_base)
    cv2.imwrite(mask_dir + 'mask_brows.png', brows_base)
    cv2.imwrite(mask_dir + 'mask_lids.png', lids_base)

    return mask_dir + 'mask.png', mask_dir + 'mask_face.png', mask_dir + 'mask_eyes.png', mask_dir + 'mask_brows.png', mask_dir + 'mask_lids.png'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MaskMaker')
    parser.add_argument('--path', default="source/for_mask/")
    parser.add_argument('--filename', default="source/")
    args = parser.parse_args()
    
    if not os.path.isdir(args.path):
            os.makedirs(args.path, exist_ok=True)

    precision_facial_masks(args.path + os.listdir(args.path)[0], args.filename, '../landmark_model/shape_predictor_68_face_landmarks.dat')
