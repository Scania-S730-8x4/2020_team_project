def eyes_cropper(indexedFaceNames, eyeDir, paddingFlag):
    # 'indexedFaceNames'는 총 분량으로부터 할당받아 온 얼굴 파일들의 이름과, 그 할당분의 순번(index), 'eyeDir'은 얼굴에서 다시 눈 부분만 자른 것을 저장할 경로.
    # 'paddingFlag'는 눈만 자른 길쭉한 형태의 image를 정사각형 형태가 되도록 검은 픽셀로 채우는 작업을 추가로 수행할지의 여부.('0 or 1 = False or True')
    from tqdm import tqdm
    import numpy as np
    import time
    import cv2

    coreNum, faceNames = indexedFaceNames
    # 'indexedFaceNames'는 멀티프로세싱을 위해 총 작업량으로부터 분할받은 부분과 그 index로,
    # index 부분을 'coreNum'으로, 할당받은 자료 부분을 'faceNames'로 각각 Un-packaging.
    # 'coreNum'이라고 해서 실제 CPU 내에서 해당 Core의 물리적인 할당 번호를 의미하는 것은 아니다!!
    # 중복의 방지와 진행률 표시 시의 구분을 위한 단순한 코드 상의 넘버링일 뿐이다.

    timeFlag = time.strftime('%m%d-%H%M%S', time.localtime(time.time()))  # 모듈 호출 당시의 시각을 기억해 둠.(파일명의 중복 방지에 활용하기 위함)
    eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')   # 눈을 검출해 내는 분류기이며, 보다 쉬운 활용을 위해 변수로 먼저 선언해 둔다.
    cnt = 0   # 눈만을 분류해 내 파일로 저장한 횟수를 계수할 변수를 0으로 초기화하며 선언.

    for faceName in tqdm(faceNames, desc="@Core {} - Cropping Eyes From Face".format(coreNum)):
        # 파일명들의 목록으로 이뤄진 'faceNames' 객체를 대상으로, 그 내용을 하나씩 호출해 처리할 것이다. 그리고 총 작업량 대비 실시간 진행률을 'tqdm'라이브러리 기능을 활용해 %로 표시한다.
        try:
            # 파일을 처리하는 도중 오류가 발생하면 전체 과정이 중단되므로, 몇몇 파일은 처리에 실패하더라도 작업이 왼료된 파일은 출력해내기 위해
            # 개별 파일 처리 중 발생하는 오류는 무시하고 진행하도록 하였다. 오류가 발생하면 해당하는 파일은 아예 출력되지 않으므로, 전체적으로 치명적 지장은 없다.
            faceFile = cv2.imread(faceName)  # 호출된 하나의 파일 경로에 해당하는 것을 불러와 변수에 대입한다.
            faceGray = cv2.cvtColor(faceFile, cv2.COLOR_BGR2GRAY)  # 눈 인식을 수행하기 전, 불러온 image를 무채색(Gray Scale)으로 변환한다.
            faceRes = int(np.shape(faceFile)[0])  # 불러온 사진의 해상도를 변수화해 둔다.(단, 불러온 사진들은 형태가 정사각형으로 미리 정형화돼있다는 것이 전제여야 한다.)

            if faceName[:1] == 0:
                # 읽어들여 올 파일의 이름에서, 맨 앞의 숫자 한 자리가 의미하는 것은 '사진 총 면적 대비 인물 부분이 차지하는 비율'이도록 설정돼 있다.
                # '0'이면 얼굴 부분만이 보다 확대된 사진으로, 전체 픽셀 대비 눈 부분의 비율이 보다 높을 것이고,
                # '1'이면 얼굴 윤곽 전체 뿐만 아니라 어깨 부분까지 나오도록 한 사진으로, 눈 부분이 차지하는 전체 대비 비율이 상대적으로 낮을 것이다.
                # 그에 따라, 불러오는 각각의 파일 별 내용의 차이가 있더라도, 안정적으로 눈을 잘 탐지해낼 수 있도록 탐지 범위를 비율 기준으로 설정해주는 부분이다.
                minSizeArg, maxSizeArg = int(faceRes * 0.166), int(faceRes * 0.2344)
            else:
                minSizeArg, maxSizeArg = int(faceRes * 0.075), int(faceRes * 0.1)

            dtcdEyes = eyeCascade.detectMultiScale(
                faceGray, scaleFactor=1.3, minSize=(minSizeArg, minSizeArg), maxSize=(maxSizeArg, maxSizeArg))
                # 제공된 image와 parameter값들을 가지고 눈 부분들을 탐지해낸 결과물을 'dtcdEyes' 객체로 반환한다.
            _, y, _, h = dtcdEyes[1]  # 여러 탐지값들 중 하나만에서, 해당 부분의 y축 시작점 값과 그 높이인 h를 선택해 가져온다.

            croppedEye = faceFile[y:y + h, :]  # 너비는 그대로 둔 채, 높이(y축)를 기준으로 눈인 부분만을 잘라내 둔다.
            eyeFileName = '{}Eyes_{}_{}_{}_{}.png'.format(eyeDir, timeFlag, coreNum, faceRes, str(cnt))
            # 잘라내 온 눈 부분의 파일명을 지정된 형식으로 설정한다.

            if paddingFlag == 1:
                # 만약 길쭉하게 잘린 눈 부분을 그대로 저장하는 게 아니라, 이미지의 잘려나간 나머지 부분을 검은 픽셀로 채워
                # 다시 정사각 형태로 만들어 저장한다는 선택지를 취했을 때('paddingFlag' Parameter의 값이 '1'일 때), 이 부분이 수행된다.
                # 눈 부분이 원래 이미지에서 차지하던 위치가 그대로 유지된 채 Padding이 수행된다.
                imgH, imgW, imgCh = faceFile.shape   # 눈 부분을 취해 온 원래 파일의 Shape 정보를 가져온다.(가로, 세로 크기 및 채널 등)
                upperDummyBase = np.zeros((imgH - (h + y), imgW, imgCh), dtype=np.uint8)   # 눈 부분의 윗쪽을 채울 더미 픽셀의 크기를 설정한다.
                lowerDummyBase = np.zeros((y, imgW, imgCh), dtype=np.uint8)   # 눈 부분의 아랫쪽을 채울 더미 픽셀의 크기를 설정한다.
                croppedEye = np.append(lowerDummyBase, croppedEye, axis=0)   # 설정한 더미 픽셀 부분들을 눈 부분과 합친다.
                croppedEye = np.append(croppedEye, upperDummyBase, axis=0)
                eyeFileName = '{}EyesSQR_{}_{}_{}_{}.png'.format(eyeDir, timeFlag, coreNum, faceRes, str(cnt))
                # 작업한 내용이 저장될 파일의 이름을 지정된 형식으로 설정한다.

            cv2.imwrite(eyeFileName, croppedEye)  # 작업 내용을 파일로 출력해 저장한다.
            cnt += 1   # 처리된 파일의 수에 하나를 더한다.
        except Exception as e:
            continue  # 하나의 파일에서 에러가 발생하더라도, 이를 무시한 채 전체 동작은 멈추지 않고 진행된다.

    return cnt   # 작업이 성공적으로 수행돼 파일로의 출력이 완료된 총 횟수를 호출측에 반환한다.
