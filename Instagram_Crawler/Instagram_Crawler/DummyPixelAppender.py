def dummy_pixel_appender(srcDir, dstDir, paddingFlag):
    # 'srcDir'은 원본 파일(들)의 경로, 'dstDir' 처리된 파일(들)이 저장될 경로이며,
    # 'paddingFlag'는 더미 픽셀을 붙일 위치 결정('0' : 눈 Image의 윗부분 / '1' : 눈 Image의 아랫부분)에 대한 Parameter이다.
    # 위치에 대한 선택에 '원본에서의 원래 위치 복원'이 없는 이유는, 해당 눈 부분을 Crop해 가져올 때 그 위치에 대한 정보까지는 가져오지 않기에,
    # 본 'Dummy Pixel Appender' Module 단독으로는 눈 부분의 원래 위치를 알아낼 방법이 전혀 없기 때문이다.
    # 다만, 본 Module에서 눈의 원래 위치를 복원할 수 없는 대신, 얼굴 전체 Image로부터 눈 부분만을 Crop해 주는 다른 Module 내에서,
    # 이미 해당 기능을 부가기능으로써 제공하고 있다. 그렇기 때문에, 굳이 해당 기능을 별도 Module에 중복으로 구현해 놓을 필요까지는 없었다.
    from tqdm import tqdm
    import numpy as np
    import cv2
    import os

    fileNames = os.listdir(srcDir)  # 원본 파일 경로 내에 존재하는 파일들의 이름 목록을 변수에 할당한다.
    cnt = 0  # 처리된 파일을 계수할 변수를 0을 초기값으로 선언한다.

    for fileName in tqdm(fileNames, desc='Padding Dummy Pixels to Image'):
        # 원본 경로 내의 파일(들)에 대해 순차적으로 처리과정을 수행하며, 전체 작업량에 대한 현재 진행률을 'tqdm'을 통해 %로 표시한다.
        faceFile = cv2.imread(srcDir + fileName)  # 경로 + 파일명을 조합하여 openCV Library의 'imread' 메소드를 이용해 해당 파일을 읽어들인다.
        imgY, imgX, imgCh = faceFile.shape  # 불러들인 Image의 높이와 너비, 채널 등의 차원 정보를 추출해 낸다.
        dummyPixel = np.zeros((imgX - imgY, imgX, imgCh), dtype=np.uint8)
        # 눈 부분만 잘라내기 전 온전한 파일의 해상도는 눈 부분만을 가져온 Image의 '너비 : X' 값에 해당한다.
        # 앞서 추출해 낸 Image의 차원 정보를 통해, 눈 부분과 합쳐 정사각형 형태가 되도록 할 흑색 Dummy Pixel 부분을 구성한다.

        if paddingFlag == 0:  # 'paddingFlag'값에 따라, 앞에서 구성한 Dummy Pixel을 원본의 위 혹은 아래에 결합할 것이다.
            padded = np.append(dummyPixel, faceFile, axis=0)  # 'paddingFlag'값이 '0'이라면 원본 Image의 위에 결합한다.
        else:
            padded = np.append(faceFile, dummyPixel, axis=0)  # 'paddingFlag'값이 '1'이라면 원본 Image의 아래에 결합한다.

        cv2.imwrite('{}{}_Padded.png'.format(dstDir, fileName), padded)
        # 처리를 거친 Image를 지정된 형식에 따라 다시 이름붙여 처리 후 파일 저장 경로에 저장한다.
        cnt += 1  # 처리된 파일 수를 1 증가시킨다.

    return cnt  # 모든 대상 파일들에 대한 처리과정이 종료되고 나면, 그 총 수를 호출측에 반환한다.
