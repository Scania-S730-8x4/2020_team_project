from Instagram_Crawler.nVidiaFaceCrawler import get_fake_face as gff
from Instagram_Crawler.InstaImageCrawler import insta_image_crawler_main as ic
from Instagram_Crawler.AlignedFaceCropper import aligned_face_cropper as afc
from Instagram_Crawler.DummyPixelAppender import dummy_pixel_appender as dpa
from Instagram_Crawler.EyesCropper import eyes_cropper as ec
from Instagram_Crawler.ReNamer import file_renamer as fr
import multiprocessing as mp
import numpy as np
import parmap
import time
import os

# Multi-Processing이 적용된 메소드에서, 활용할 코어 갯수의 기본값을 설정한다. 기본값은 해당 PC CPU 코어 수(논리코어 포함)의 1/2이다.
# 기본값을 Core 수의 절반으로 설정한 이유는, Core 하나만으로는 버거운 Process도 더러 존재하는데, 그런 것을 모든 Core에다 각각 할당하면,
# 과부하로 인해 CPU의 Throttling이나 System Shutdown이 일어날 가능성이 있기 때문이다.
cores = int(mp.cpu_count() / 2)


# nVidia의 초상권이 없는 Fake Face를 무한정 수집하는 메소드. [[[[ 본 메소드는 Multi-Processing으로 수행 시 중복된 사진을 가져오므로, Multi-Processing을 지원하지 않는다. ]]]]
def nVidia_crawler():
    saveDir = 'D:/Downloaded/nVidiaFace/'  # 가져온 Image들을 저장할 경로를 설정한다.
    if os.path.isdir(saveDir) is not True:
        os.makedirs(saveDir)
        # 경로에 해당하는 폴더가 존재하지 않는 것이 하나라도 있을 경우, 자동으로 해당 경로 전체에 해당하는 모든 폴더를 생성한다.
        # 이미 존재하는 폴더에 대해서는 오류 발생을 무시하고 과정을 진행한다.
    gff(saveDir)  # 저장 경로를 Parameter로 넘겨주면, 해당 Module내에서 모든 동작이 반복/지속적으로 수행된다. 중단을 원할 시 강제 종료가 필요하다.


# Instagram에서, 입력한 #Tag에 해당하는 Image들을 스크롤 다운해가며 수집해 가져오는 메소드. [[[[ Multi-Processing 지원 ]]]]
def insta_crawler():
    baseUrl = 'https://www.instagram.com/explore/tags/'  # 인스타그램 검색 페이지에 대한 기본 접속 Url이다.
    tags = input('검색할 태그 입력 [다수 입력 시 콤마(",")로 구분] : ').replace(" ", "").split(",")  # Tag를 입력받는다. 복수 입력이 가능하다.
    # 공백이 입력되거나, ",,,,,"등으로 콤마만 입력될 경우, 오류는 발생하지 않지만 최소 1회/최대 "콤마 갯수+1"회 만큼 헛돈 후, Image의 저장이 없이 종료된다.
    urls = []  # 개별 #Tag에 대한 실제 검색 수행 Url(들)을 담아 둘 배열('Urls')을 선언한다.

    for tag in tags:
        urls.append(baseUrl + tag + '/')  # '#Tag로 검색'의 실제 수행에 해당하는 Url(들)을 먼저 조합해 둔다.

    tagAndUrls = list(zip(tags, urls))  # #Tag와 그것을 이용해 조합한 검색 Url을 엮어 배열에 담아 둔다.

    while True:  # 스크롤 다운을 반복할 횟수는 자연수만 입력받도록 제한하였다.
        iteration = int(input('스크롤을 반복할 횟수 지정 [0 초과 자연수만 입력하세요!] : '))
        if iteration > 0:
            break
        else:
            print("\nInvalid input : Natural Number ONLY!! Try again.\n")
            continue

    imgDir = 'D:/Downloaded/InstaIMG/'  # 가져온 Image를 파일로 저장할 대상 경로를 지정한다.
    if os.path.isdir(imgDir) is not True:
        os.makedirs(imgDir, exist_ok=True)
        # 경로에 해당하는 폴더가 존재하지 않는 것이 하나라도 있을 경우, 자동으로 해당 경로 전체에 해당하는 모든 폴더를 생성한다.
        # 이미 존재하는 폴더에 대해서는 오류 발생을 무시하고 과정을 진행한다.
    _, cnt = ic(tagAndUrls, imgDir, iteration)
    # 해당 기능을 수행하는 Module에 Parameter로 #Tag에 대한 Url(들), 저장 경로, 스크롤 다운 반복 횟수를 넘겨주고, 가져온 Image의 총 수를 반환받는다.
    print("Total Images Saved : {}\n".format(cnt))  # 가져온 Image의 총 수에 대한 안내 메시지를 출력한 후, 메소드의 Process를 종료한다.


# 위의 단순 Instagram Image Crawler에 여러 Pre-Processing 기능을 결합한 복합 메소드이다. [[[[ Multi-Processing 지원 ]]]]
def insta_aligned_face_crawler():
    baseUrl = 'https://www.instagram.com/explore/tags/'  # 인스타그램 검색 페이지에 대한 기본 접속 Url이다.
    tags = input('검색할 태그 입력 [다수 입력 시 콤마(",")로 구분] : ').replace(" ", "").split(",")  # Tag를 입력받는다. 복수 입력이 가능하다.
    # 공백이 입력되거나, ",,,,,"등으로 콤마만 입력될 경우, 오류는 발생하지 않지만 최소 1회/최대 "콤마 갯수+1"회 만큼 헛돈 후, Image의 저장이 없이 종료된다.
    interval, paddingFlag, resFlag, scope, flag = 0, 0, 0, 0, 1  # 이후 옵션 선택에서 Parameter 값들로 활용될 변수들을 초기화하며 선언한다.
    urls, imgRes = [], []
    # 개별 #Tag에 대한 실제 검색 수행 Url(들)을 담아 둘 배열('Urls')과,
    # 가져온 사진들에서 얼굴 부분만을 추려 저장할 때 출력할 해상도값(들)을 담아 둘 배열('imgRes')을 각각 선언한다.

    for tag in tags:
        urls.append(baseUrl + tag + '/')  # '#Tag로 검색'의 실제 수행에 해당하는 Url(들)을 먼저 조합해 둔다.

    tagAndUrls = list(zip(tags, urls))  # #Tag와 그것을 이용해 조합한 검색 Url을 엮어 배열에 담아 둔다.

    while True:  # 스크롤 다운을 반복할 횟수는 자연수만 입력받도록 제한하였다.
        iteration = int(input('스크롤을 반복할 횟수 지정 [0 초과 자연수만 입력하세요!] : '))
        if iteration > 0:
            break
        else:
            print("\nInvalid input : Natural Number ONLY!! Try again.\n")
            continue

    while resFlag == 0:  # 정렬해 Crop한 얼굴 이미지의 출력 해상도를 선택할 수 있도록 한다. 복수 선택 가능.
        imgResFlags = input(
            '얼굴 이미지의 출력 해상도(Pixel) 지정\n[복수입력 가능 >> 0 : 256*256 / 1 : 512*512 / 2 : 1024*1024]\n복수 선택 시 콤마(,)로 구분 : ').replace(
            " ", "").split(",")
        resFlag = 1
        for imgResFlag in imgResFlags:
            if int(imgResFlag) in range(0, 3):
                if imgResFlag == '0':
                    imgRes.append(256)
                elif imgResFlag == '1':
                    imgRes.append(512)
                else:
                    imgRes.append(1024)
            else:  # 선택지에 해당하는 값 이외의 입력에 대해서는 주의 메시지를 띄운 후 다시 입력하도록 만든다.
                print("[ 0, 1, 2 ] ONLY!!! Try Again.")
                resFlag = 0
                break

    while True:   # 얼굴을 잘라 가져올 때, 대상 인물의 신체 범위를 지정
        scopeFlag = int(input('이미지에 나타낼 신체 범위 지정 [0 : 이목구비 확대 / 1 : 머리 전체 및 어깨 상단] : '))
        if scopeFlag in range(0, 2):
            if scopeFlag == 0:
                scope = 0.1
            else:
                scope = 1
            break
        else:  # 선택지에 해당하는 값 이외의 입력에 대해서는 주의 메시지를 띄운 후 다시 입력하도록 만든다.
            print("[ 0, 1 ] ONLY!!! Try Again.")
            continue

    while True:  # 잘라온 얼굴 부분 이미지에서, 다시금 눈 부분만을 잘라 가져올지 선택할 수 있다.
        eyeFlag = int(input('얼굴에서 눈 부분만을 따로 Crop합니다 [0 : 아니오 / 1 : 예] : '))
        if eyeFlag == 1 or eyeFlag == 0:
            if eyeFlag == 1:
                while True:  # 눈 부분만 잘라 가져온다고 했을 때, 잘려나간 나머지 부분을 검은 Pixel로 채워 정사각형 형태를 유지할지 선택할 수 있다.
                    paddingFlag = int(input('눈을 제외한 나머지 부분을 Padding합니다 [0 : 아니오 / 1 : 예] : '))
                    if paddingFlag == 1 or paddingFlag == 0:
                        break
                    else:  # 선택지에 해당하는 값 이외의 입력에 대해서는 주의 메시지를 띄운 후 다시 입력하도록 만든다.
                        print("[ 0 or 1 ] ONLY!!! Try Again.")
            break
        else:  # 선택지에 해당하는 값 이외의 입력에 대해서는 주의 메시지를 띄운 후 다시 입력하도록 만든다.
            print("[ 0 or 1 ] ONLY!!! Try Again.")

    while True:  # 일회성으로 동작할지, 작동 개시 후 자동으로 주기적으로 동작할지를 선택할 수 있다.
        autoFlag = int(input('수동 정지 시까지 자동으로 단위 시간마다 Crawling을 실시합니다. [0 : 한 번만 작동 / 1 : 자동 지속] : '))
        if autoFlag == 1 or autoFlag == 0:
            if autoFlag == 1:
                while True:  # 각 입력 값에 따라 작동 시간의 텀을 선택할 수 있다. 복수입력은 불가.
                    intervalSwitch = int(input('Crawling 간격(단위시간) [0 : 3시간 / 1 : 6시간 / 2 : 12시간 / 3 : 24시간] : '))
                    if intervalSwitch in range(0, 4):
                        if intervalSwitch == 0:
                            interval = 3
                        elif intervalSwitch == 1:
                            interval = 6
                        elif intervalSwitch == 2:
                            interval = 12
                        else:
                            interval = 24
                        break
                    else:  # 선택지에 해당하는 값 이외의 입력에 대해서는 주의 메시지를 띄운 후 다시 입력하도록 만든다.
                        print("[ 0, 1, 2 ] ONLY!!! Try Again.")
            break
        else:  # 선택지에 해당하는 값 이외의 입력에 대해서는 주의 메시지를 띄운 후 다시 입력하도록 만든다.
            print("[ 0 or 1 ] ONLY!!! Try Again.")

    while flag != 0:  # 일회성 동작과 자동 주기 동작을 'flag'의 값에 따라 구별해 수행한다.
        imgDir = 'D:/Downloaded/InstaIMG/'  # Intragram에서 가져온 Raw-Image들을 저장할 경로를 지정한다.
        if os.path.isdir(imgDir) is not True:
            os.makedirs(imgDir, exist_ok=True)
            # 경로에 해당하는 폴더가 존재하지 않는 것이 하나라도 있을 경우, 자동으로 해당 경로 전체에 해당하는 모든 폴더를 생성한다.
            # 이미 존재하는 폴더에 대해서는 오류 발생을 무시하고 과정을 진행한다.
        fileNames, _ = ic(tagAndUrls, imgDir, iteration)

        crpFileDir = 'D:/Downloaded/InstaFace/'  # 얼굴 부분만 취한 Image를 저장할 경로를 지정한다.
        if os.path.isdir(crpFileDir) is not True:
            os.makedirs(crpFileDir, exist_ok=True)
            # 경로에 해당하는 폴더가 존재하지 않는 것이 하나라도 있을 경우, 자동으로 해당 경로 전체에 해당하는 모든 폴더를 생성한다.
            # 이미 존재하는 폴더에 대해서는 오류 발생을 무시하고 과정을 진행한다.
        for res in imgRes:
            dir = '{}{}/'.format(crpFileDir, str(res))
            if os.path.isdir(dir) is not True:
                os.mkdir(dir)
                # 출력할 해상도에 해당하는 폴더가 존재하지 않을 경우, 해당 폴더를 생성한다.
        indexedFileNames = list(enumerate(np.array_split(fileNames, cores)))
        # Multi-Processing을 위해, 각각의 병렬 Process에 할당할 작업 분량을 CPU Core수에 맞춰 고르게 분할해 배열화해 둔다.
        faceNames = []  # 얼굴 부분만 잘라 가져온 Image들의 파일명을 적재할 배열을 선언한다.
        cnt = 0  # 해당 작업을 통해 저장된 파일들을 계수할 변수를 초기값 0으로 선언한다.

        for faceNamesPerCore, cntPerCore in parmap.map(afc, indexedFileNames, crpFileDir, imgRes, scope):
            # 각각의 병렬 Process에게 파일명 목록, 작업한 파일을 저장 경로, 저장할 해상도, 신체 범위를 Parameter로 넘겨준 후,
            # 얼굴 부분만 잘라 온 파일들의 파일명 목록과 그 총 수를 반환받는다.
            for faceName in faceNamesPerCore:
                faceNames.append(faceName)
                # 반환받은 파일명들을 전체 파일명 목록에 더해 넣는다.
            cnt += cntPerCore  # 개별 Process가 작업을 완료한 파일들의 수를 모두 더한다.

        print("Total Face Cropped Count : {}\n".format(cnt))  # 병렬 작업이 모두 완료된 후, 얼굴만 가져와 저장한 Image의 수를 안내한다.
        for fileName in fileNames:
            os.remove(fileName)  # Instagram에서 가져온 최초의 Raw-Image들은 모두 삭제한다.

        if eyeFlag == 1:  # 얼굴을 잘라 온 후, 추가적으로 다시 눈 부분만 추려낼 것을 선택했을 시 수행되는 작업이다.
            cnt = 0  # 작업한 파일 수를 다시 0으로 초기화한다.
            eyeDir = 'D:/Downloaded/InstaEyes/'  # 눈 부분만을 취한 Image를 따로 저장할 경로를 지정한다.
            if os.path.isdir(eyeDir) is not True:
                os.makedirs(eyeDir, exist_ok=True)
                # 경로에 해당하는 폴더가 존재하지 않는 것이 하나라도 있을 경우, 자동으로 해당 경로 전체에 해당하는 모든 폴더를 생성한다.
                # 이미 존재하는 폴더에 대해서는 오류 발생을 무시하고 과정을 진행한다.
            indexedFaceNames = list(enumerate(np.array_split(faceNames, cores)))
            # Multi-Processing을 위해, 각각의 병렬 Process에 할당할 작업 분량을 CPU Core수에 맞춰 고르게 분할해 배열화해 둔다.

            for cntPerCore in parmap.map(ec, indexedFaceNames, eyeDir, paddingFlag):
                cnt += cntPerCore
                # 각각의 병렬 Process에게 파일명 목록, 작업한 파일을 저장 경로, Padding 여부를 Parameter로 넘겨준 후,
                # 작업 완료해 저장된 파일의 총 수를 반환받는다.
            print("Total Eyes Cropped Count : {}\n".format(cnt))  # 작업 완료 파일 총 수를 메시지로 안내한다.

        flag = autoFlag  # 최초 작업 Cycle의 수행이 완료되면, 'flag'값을 이후 자동 작동 선택 여부에 따라 다르게 변경한다.
        if flag == 1:  # 지속 작동을 선택한 경우라면, 다음 번 작동까지의 시간 텀을 메시지로 안내하도록 한다.
            print("\n{}시간 후, 지정한 내용의 Crawling이 다시 수행됩니다.\n".format(interval))
        time.sleep(interval * 3600)


# Raw Image Data로부터 얼굴 부분을 탐색해낸 후 바르게 정렬한 다음 잘라내 그 부분만 가져오는 기능의 메소드이다. [[[[ Multi-Processing 지원 ]]]]
def aligned_face_cropper():
    imgRes = []  # 얼굴 부분만을 추려 저장할 때 출력할 해상도값(들)을 담아 둘 배열을 선언한다.
    flag, cnt, scope = 0, 0, 0  # 이후 옵션 선택에서 Parameter 값들로 활용될 변수들을 초기화하며 선언한다.

    while flag == 0:  # 정렬해 Crop한 얼굴 이미지의 출력 해상도를 선택할 수 있도록 한다. 복수 선택 가능.
        imgResFlags = input(
            '얼굴 이미지의 출력 해상도(Pixel) 지정\n[복수입력 가능 >> 0 : 256*256 / 1 : 512*512 / 2 : 1024*1024]\n복수 선택 시 콤마(,)로 구분 : '
            ).replace(" ", "").split(",")
        flag = 1
        for imgResFlag in imgResFlags:
            if int(imgResFlag) in range(0, 3):
                if imgResFlag == '0':
                    imgRes.append(256)
                elif imgResFlag == '1':
                    imgRes.append(512)
                else:
                    imgRes.append(1024)
            else:  # 선택지에 해당하는 값 이외의 입력에 대해서는 주의 메시지를 띄운 후 다시 입력하도록 만든다.
                print("[ 0, 1, 2 ] ONLY!!! Try Again.")
                flag = 0
                break

    while True:   # 얼굴을 잘라 가져올 때, 대상 인물의 신체 범위를 지정
        scopeFlag = int(input('이미지에 나타낼 범위를 지정 [0 : 이목구비 확대 / 1 : 머리 전체 및 어깨 상단] : '))
        if scopeFlag in range(0, 2):
            if scopeFlag == 0:
                scope = 0.1
            else:
                scope = 1
            break
        else:  # 선택지에 해당하는 값 이외의 입력에 대해서는 주의 메시지를 띄운 후 다시 입력하도록 만든다.
            print("[ 0, 1 ] ONLY!!! Try Again.")
            continue

    imgDir = "D:/Downloaded/InstaIMG/"  # Intragram에서 가져온 Raw-Image들을 저장할 경로를 지정한다.
    if os.path.isdir(imgDir) is not True:
        os.makedirs(imgDir, exist_ok=True)
        # 경로에 해당하는 폴더가 존재하지 않는 것이 하나라도 있을 경우, 자동으로 해당 경로 전체에 해당하는 모든 폴더를 생성한다.
        # 이미 존재하는 폴더에 대해서는 오류 발생을 무시하고 과정을 진행한다.
    fileNames = []  # 저장한 파일들의 파일명 목록을 반환받을 배열을 선언한다.
    for name in os.listdir(imgDir):
        fileNames.append(imgDir + name)  # 반환받은 파일명과 경로명을 결합해 배열에 적재한다.

    crpFileDir = 'D:/Downloaded/InstaFace/'  # 얼굴 부분만 취한 Image를 저장할 경로를 지정한다.
    if os.path.isdir(crpFileDir) is not True:
        os.makedirs(crpFileDir, exist_ok=True)
        # 경로에 해당하는 폴더가 존재하지 않는 것이 하나라도 있을 경우, 자동으로 해당 경로 전체에 해당하는 모든 폴더를 생성한다.
        # 이미 존재하는 폴더에 대해서는 오류 발생을 무시하고 과정을 진행한다.
    for res in imgRes:
        dir = '{}{}/'.format(crpFileDir, str(res))
        if os.path.isdir(dir) is not True:
            os.mkdir(dir)
            # 출력할 해상도에 해당하는 폴더가 존재하지 않을 경우, 해당 폴더를 생성한다.
    indexedFileNames = list(enumerate(np.array_split(fileNames, cores)))
    # Multi-Processing을 위해, 각각의 병렬 Process에 할당할 작업 분량을 CPU Core수에 맞춰 고르게 분할해 배열화해 둔다.

    for _, cntPerCore in parmap.map(afc, indexedFileNames, crpFileDir, imgRes, scope):
        # 각각의 병렬 Process에게 파일명 목록, 작업한 파일을 저장 경로, 저장할 해상도, 신체 범위를 Parameter로 넘겨준 후,
        # 얼굴 부분만 잘라 온 파일들의 총 수를 반환받는다.
        cnt += cntPerCore  # 각각의 Process로부터 반환받은 수치를 합산한다.

    # for fileName in fileNames:
    #     os.remove(fileName)

    print("Total Face Cropped Count : {}\n".format(cnt))   # 합산한 총 파일 수를 메시지로 안내한다.


# 정렬해 잘라 가져온 얼굴 Image로부터 다시 두 눈 부분만을 잘라내 별도로 저장해 주는 기능의 메소드. [[[[ Multi-Processing 지원 ]]]]
def eyes_cropper():
    imgRes = 256
    while True:  # Crop할 대상 얼굴 이미지의 해상도에 해당하는 폴더를 선택할 수 있도록 한다.
        imgResFlag = int(input('대상 얼굴 이미지 해상도(Pixel) 폴더 선택 [0 : 256*256 / 1 : 512*512 / 2 : 1024*1024] : '))
        if imgResFlag in range(0, 3):
            if imgResFlag == 0:
                imgRes = 256
            elif imgResFlag == 1:
                imgRes = 512
            else:
                imgRes = 1024
            break
        else:  # 선택지에 해당하는 값 이외의 입력에 대해서는 주의 메시지를 띄운 후 다시 입력하도록 만든다.
            print("[ 0, 1, 2 ] ONLY!!! Try Again.")
            continue

    srcDir = "D:/Downloaded/InstaFace/{}/".format(imgRes)  # 앞서 입력받은 해상도와 파일 경로를 결합한다.
    faceNames = []  # 대상 폴더 내에 존재하는 파일들의 이름을 저장해 둘 배열을 선언한다.
    for name in os.listdir(srcDir):
        faceNames.append(srcDir + name)  # 앞서 확립해 둔, 작업 대상에 해당하는 해상도의 파일 경로와 그 파일명들을 결합해 저장해 둔다.

    eyeDir = 'D:/Downloaded/InstaEyes/'  # 눈 부분만을 취한 Image 파일을 저장할 경로를 설정한다.
    if os.path.isdir(eyeDir) is not True:
        os.makedirs(eyeDir, exist_ok=True)
        # 경로에 해당하는 폴더가 존재하지 않는 것이 하나라도 있을 경우, 자동으로 해당 경로 전체에 해당하는 모든 폴더를 생성한다.
        # 이미 존재하는 폴더에 대해서는 오류 발생을 무시하고 과정을 진행한다.
    cnt = 0  # 작업을 완료한 파일 수를 기억할 변수를 초기값을 0으로 하여 선언한다.

    while True:  # 눈 부분만 잘라 가져온다고 했을 때, 잘려나간 나머지 부분을 검은 Pixel로 채워 정사각형 형태를 유지할지 선택할 수 있다.
        paddingFlag = int(input('눈을 제외한 나머지 부분을 Padding합니다 [0 : 아니오 / 1 : 예] : '))
        if paddingFlag == 1 or paddingFlag == 0:
            break
        else:  # 선택지에 해당하는 값 이외의 입력에 대해서는 주의 메시지를 띄운 후 다시 입력하도록 만든다.
            print("[ 0 or 1 ] ONLY!!! Try Again.")

    indexedFaceNames = list(enumerate(np.array_split(faceNames, cores)))
    # Multi-Processing을 위해, 각각의 병렬 Process에 할당할 작업 분량을 CPU Core수에 맞춰 고르게 분할해 배열화해 둔다.

    for cntPerCore in parmap.map(ec, indexedFaceNames, eyeDir, paddingFlag):
        # 각각의 병렬 Process에게 파일명 목록, 작업한 파일을 저장 경로, 저장할 해상도, 신체 범위를 Parameter로 넘겨준 후,
        # 얼굴 부분만 잘라 온 파일들의 파일명 목록과 그 총 수를 반환받는다.
        cnt += cntPerCore  # 각각의 Process로부터 반환받은 수치를 합산한다.

    print("Total Eyes Cropped Count : {}\n".format(cnt))  # 합산한 총 파일 수를 메시지로 출력해 안내한다.


# 눈 부분만 잘라 가져온 Image가 어떠한 이유로 정사각형 형태의 Image가 돼야 할 때, 검은 Dummy Pixel을 붙여 정사각형화 해 주는 메소드.
# [[[[ Multi-Processing을 지원하지 않음. ]]]]
def dummy_pixel_appender():
    srcDir = 'D:/Downloaded/InstaEyes/'   # 눈을 잘라 올 대상인 원본 얼굴 파일들을 불러올 경로를 지정한다.
    dstDir = 'D:/Downloaded/InstaEyesSquared/'   # 눈만 잘라 온 Image를 파일로 저장할 경로를 지정한다.
    if os.path.isdir(dstDir) is not True:
        os.makedirs(dstDir, exist_ok=True)
        # 경로에 해당하는 폴더가 존재하지 않는 것이 하나라도 있을 경우, 자동으로 해당 경로 전체에 해당하는 모든 폴더를 생성한다.
        # 이미 존재하는 폴더에 대해서는 오류 발생을 무시하고 과정을 진행한다.

    while True:  # 지정된 값을 입력하여 Padding할 부분을 설정한다.
        paddingFlag = int(input('Padding할 부분을 설정합니다 [0 : 원본 이미지의 위 / 1 : 원본 이미지의 아래] : '))
        if paddingFlag == 1 or paddingFlag == 0:
            break
        else:   # 선택지에 해당하는 값 이외의 입력에 대해서는 주의 메시지를 띄운 후 다시 입력하도록 만든다.
            print("[ 0 or 1 ] ONLY!!! Try Again.")

    cnt = dpa(srcDir, dstDir, paddingFlag)
    # 실제 작업을 수행할 메소드에게 원본들의 경로, 작업 완료 파일 저장 경로, 패딩 위치 정보를 Parameter로 넘겨주고,
    # 작업을 완료한 횟수를 반환받는다.

    print("Total Squared Image Count : {}\n".format(cnt))  # 총 작업 완료 횟수를 메시지로 출력해 안내한다.


# 지정한 특정 경로 내의 파일들을 재명명하여 동일하거나 특정한 다른 경로에 다시 저장해 주는 메소드. [[[[ Multi-Processing을 지원하지 않음. ]]]]
def renamer():
    srcDir = input('Input Source Directory [ex: D:/..../..../ ] >> ')  # 이름을 바꿀 원본 파일들을 불러올 경로를 지정한다.
    dstDir = input('Input Target Directory [ex: D:/..../..../ ] >> ')  # 이름을 바꾼 파일들을 저장할 경로를 지정한다.
    if os.path.isdir(dstDir) is not True:
        os.makedirs(dstDir, exist_ok=True)
        # 경로에 해당하는 폴더가 존재하지 않는 것이 하나라도 있을 경우, 자동으로 해당 경로 전체에 해당하는 모든 폴더를 생성한다.
        # 이미 존재하는 폴더에 대해서는 오류 발생을 무시하고 과정을 진행한다.
    fr(srcDir, dstDir)  # 실제 작업을 수행할 메소드에게 원본 대상 경로, 작업 완료 파일 저장 경로를 Parameter로 넘겨준다.


# 작동 시 가장 먼저 구동되는 Main 메소드이다. 여러 기능을 선택할 수 있으며, 순차적인 선택으로 Process들을 Pipeline과 유사한 형태로 동작시킬 수도 있다.
def main():
    runFlag = 1
    while runFlag == 1:  # 'runFlag'가 1이 아닌 다른 값으로 변경되면 반복 작동을 멈출 것이다.
        inputFlag = 0
        launchCode = []  # 작업을 여럿 입력받을 수 있으므로, 입력값을 담아 둘 것을 배열로 선언해 둔다.
        while inputFlag == 0:  # 작동에 필요한 올바른 값들만이 입력될 때까지 계속 과정을 반복할 것이다.
            try:
                launchCode = input('사용할 기능의 번호를 선택하세요.\n'
                                   '[다수 입력 시 콤마(",")로 구분되며, 입력한 순서대로 순차 실행됩니다.]\n\n'
                                   '1 : nVidia Fake Face Crawler\n'
                                   '2 : Instagram #Tag Image Crawler\n'
                                   '3 : Automated Instagram #Tag Pre-Processed Image Crawler\n'
                                   '4 : Dlib Aligned Face Cropper\n'
                                   '5 : openCV Eyes Cropper\n'
                                   '6 : Dummy Pixel Appender(= Image Squaring)\n'
                                   '7 : Multiple File Re-Namer\n'
                                   '0 : Exit Launcher\n\n'
                                   '위 목록에 해당하는 번호만 입력해 주십시오 : ').replace(" ", "").split(",")
                inputFlag = 1  # 우선 올바른 값들을 입력받았다는 것을 전제로, while문을 빠져나갈 수 있도록 'inputFlag'의 값을 변경한다.
                for code in launchCode:
                    if int(code) not in range(0, 10):  # 선택지에 해당하는 값 이외의 입력에 대해서는 주의 메시지를 띄운 후 다시 입력하도록 만든다.
                        print('\nERROR : Invalid Input!!! Try Again.\n\n')
                        inputFlag = 0  # 잘못된 입력이 확인됐으므로 다시 과정을 반복하도록 'inputFlag'의 값을 되돌린다.
                        break  # 하나라도 잘못 입력한 값이 있다면, 남은 값들을 더 검사하지 않고 바로 재입력 과정으로 돌아가도록 한다.
            except Exception as e:  # 선택지에 해당하는 값 이외의 입력에 대해서는 주의 메시지를 띄운 후 다시 입력하도록 만든다.
                print('\nERROR : Invalid Input!!! Try Again.\n\n')
                inputFlag = 0  # 잘못된 입력이 확인됐으므로 다시 과정을 반복하도록 'inputFlag'의 값을 되돌린다.

        for code in launchCode:  # 입력받은 값들을 확인해 각각에 해당하는 동작을 입력한 순서대로 수행한다.
            if int(code) == 1:
                nVidia_crawler()
            elif int(code) == 2:
                insta_crawler()
            elif int(code) == 3:
                insta_aligned_face_crawler()
            elif int(code) == 4:
                aligned_face_cropper()
            elif int(code) == 5:
                eyes_cropper()
            elif int(code) == 6:
                dummy_pixel_appender()
            elif int(code) == 7:
                renamer()
            else:  # '0'을 값으로 입력받으면, Process를 종료한다.
                runFlag = 0


if __name__ == '__main__':
    main()
