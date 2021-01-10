from image_2_style_gan.image_crossover_face import image_crossover_face
from image_2_style_gan.image_crossover_eyes import image_crossover_eyes
from connector.img_processing_manage.path_manager import *
from connector.result_manage.result_processing import *
import shutil


def main_processing(data, rand_uuid):
    try:
        if 'mode' in data:
            mode = data['mode']
        else:
            mode = 'eyes'

        gender = data['gender']
        process_selection = 0

        # 전송받은 데이터와 프로세스 선택 변수 넘겨주기
        BASE_DIR, RAW_DIR = origin_image_control(
            data, process_selection, rand_uuid)
        if data['custom']:
            process_selection = 1
            custom_image_control(data, process_selection, rand_uuid)

        print("\n********** Image processing succeed, send to model **********\n")
        print(f"[Condition]\nMode: {mode}\nGender: {gender}\n")
        if mode == 'eyes':
            input_image, output_image = image_crossover_eyes(
                BASE_DIR, RAW_DIR, rand_uuid, process_selection, gender)
        else:
            input_image, output_image = image_crossover_face(
                BASE_DIR, RAW_DIR, rand_uuid, process_selection, gender)
        print("\n********** Model processing succeed, post data **********\n")

        # 이미지 base64형식으로 변환
        json_data = result_processing(input_image, output_image, rand_uuid)
        shutil.rmtree(BASE_DIR)  # UUID 디렉터리 삭제
        return json_data
    except Exception as e:
        shutil.rmtree(BASE_DIR)  # UUID 디렉터리 삭제
        print(f'json_data part error: {e}')
