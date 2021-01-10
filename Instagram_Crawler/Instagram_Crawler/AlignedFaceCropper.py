def aligned_face_cropper(indexedFileNames, crpFileDir, imgRes, scope):
    # 'indexedFileNames'는 총 분량으로부터 할당받아 온 원본 파일들의 이름과, 그 할당분의 순번(index), 'crpFileDir'은 얼굴 부분만 자른 것을 저장할 경로.
    # 'imgRes'는 이미지를 저장할 해상도(다수 선택 가능)에 대한 Parameter, 'scope'는 이미지 내에서 인물 부분이 차지하는 범위를 설정하는 Parameter 값이다.
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import dlib
    import time

    cnt = 0  # 출력돼 나온 총 파일의 수를 계수할 변수를 0으로 초기화하며 선언한다.
    coreNum, fileNames = indexedFileNames
    # 'indexedFileNames'은 멀티프로세싱을 위해 총 작업량으로부터 분할받은 부분과 그 index 정보로,
    # index 부분을 'coreNum'으로, 할당받은 자료 부분을 'fileNames'로 각각 Un-packaging.
    # 'coreNum'이라고 해서 실제 CPU 내에서 해당 Core의 물리적인 할당 번호를 의미하는 것은 아니다!!
    # 중복의 방지와 진행률 표시 시의 구분을 위한 단순한 코드 상의 넘버링일 뿐이다.

    timeFlag = time.strftime('%m%d-%H%M%S', time.localtime(time.time()))   # 모듈 작동 당시의 시각을 기억해 둠.(파일명의 중복 방지에 활용하기 위함)
    shapePrd = dlib.shape_predictor('D:/HSH/Model/shape_predictor_68_face_landmarks.dat')   # 학습된 형태 예측기의 모델 파일을 불러옴.(!!!!! Github 용량 제한으로 포함돼있지 않음 !!!!!)
    faceDtcr = dlib.get_frontal_face_detector()   # 얼굴 정면을 탐지하는 dlib의 Method를 변수화.
    faceDirs = []   # 출력한 파일을 다른 함수 등에 활용할 수 있도록, 작업을 통해 출력된 파일의 이름들을 저장할 배열 선언.

    for fileName in tqdm(fileNames, desc="@Core {} - Aligning & Cropping".format(coreNum)):
        # 특정 폴더 내 사진들의 이름 리스트인 'fileNames'의 내용에 따라 이미지를 불러온 후, 컨테이너에 적재.
        if plt.imread(fileName).shape[0] < 600:
            continue   # 읽어 올 이미지의 크기가 정사각형 한 차원 기준 600픽셀 미만이라면, 작업을 진행하지 않고 다음 파일로 넘어가도록 처리하였다.
        imgLoaded = dlib.load_rgb_image(fileName)   # 파일을 RGB Color 이미지로 불러옴.
        dtcdFaces = faceDtcr(imgLoaded, 1)   # 불러온 이미지에서 얼굴(들)을 탐지해 변수에 저장.
        objects = dlib.full_object_detections()    # 탐지해 낸 랜드마크 포인트들을 담을 변수를 선언.

        for dtcdFace in dtcdFaces:  # 찾은 얼굴 부분 안에서 모델을 이용해 랜드마크 포인트들을 탐지.
            if dtcdFace.right() - dtcdFace.left() < 200:
                continue   # 탐지해 낸 얼굴 부분의 한 차원 기준 크기가 200픽셀 미만이면, 작업을 진행하지 않고 다음 파일로 넘어간다.

            shape = shapePrd(imgLoaded, dtcdFace)
            objects.append(shape)  # 찾아낸 랜드마크 포인트들의 정보를 앞서 선언해 둔 변수에 적재.

        for res in imgRes:
            # 이미지를 저장할 해상도를 복수 선택할 수 있게 해 뒀으므로, 그 선택지를 입력한 수 만큼 반복해 작동한다.
            saveCnt = 0   # 저장한 파일을 계수할 변수 선언.
            # 'cropFileDir'은 Crop해 온 파일들이 저장될 포괄적 폴더의 경로, 'res'는 그 폴더 안에 생성될 또 다른 폴더의 이름이자 저장될 해상도이다.
            dir = '{}{}/'.format(crpFileDir, str(res))
            try:
                # 탐지한 포인트 정보와 주어진 Parameter 값들을 가지고 얼굴을 바르게 정렬한 후 잘라낸 다음, 적재한다.
                # 개별 파일에 대한 작업 도중 오류가 발생할 시(얼굴을 탐지하지 못했다거나 등의 경우),
                # 그 오류로 인해 전체 과정이 중단돼버리는 대신, 오류를 무시하고 다음 파일로 그냥 넘어갈 수 있도록 Exception 처리를 하였다.
                # 즉, 특정 몇몇 파일로부터 얼굴을 찾지 못했더라도 이를 그냥 무시하고, 다른 파일에서라도 얼굴을 찾아냈다면 그 것들은 온전히 가져올 수 있도록 한 것이다.
                face = dlib.get_face_chips(imgLoaded, objects, size=res, padding=scope)
                # 앞서 추출해 낸 얼굴을 Parameter값(해상도, 인물 부분의 표현 범위)에 따라 .png 파일로 출력.
                # 동일 인물의 중복 출력을 방지하기 위해 한 사진 당 얼굴 하나씩만을 출력하도록 했다.
                fileDir = '{}{}_Aligned_{}_{}_{}_{}.png'.format(dir, int(scope), timeFlag, coreNum, res, cnt)  # 파일명을 형식에 따라 설정.
                dlib.save_image(face[0], fileDir)  # 'face'배열의 가장 첫 내용만을 지정한 파일명으로 저장.
                faceDirs.append(fileDir)  # 나중 과정에 활용할 수 있도록, 잘라내 저장한 얼굴들에 대한 파일명을 배열에 추가.
                saveCnt += 1  # 저장한 파일의 수를 1 증가시킨다.
            except Exception as e:
                break

        cnt += saveCnt
        # 'saveCnt'는 for Loop의 내부변수이기 때문에, 매 Iteration 시마다 초기화될 것이므로,
        # 회 당 산출된 값을 매 Iteration이 종료될 때마다 전역변수에 더해놓아 줘야 그 수가 의미를 상실하지 않는다.

    return faceDirs, cnt  # 출력한 파일들의 이름들 목록과 그 총 수를 호출측에 반환.
