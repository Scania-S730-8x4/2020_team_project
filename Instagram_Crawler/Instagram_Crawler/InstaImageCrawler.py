from bs4 import BeautifulSoup
from selenium import webdriver
from urllib.request import urlopen
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import parmap
import time

cores = int(mp.cpu_count())
# 프로세스가 수행되는 PC의 CPU Core 수를 확인해 변수로 저장해 둔다. 본 모듈의 작업은 작업 당 CPU 점유율이 높은 복잡한 프로세스는 아니기 때문에
# 존재하는 Core의 수 만큼 작업을 병렬로 수행하게 할 것이다.

def image_saver(indexedImgList, imgDir):
    timeFlag = time.strftime('%m%d-%H%M%S', time.localtime(time.time()))  # 모듈 호출 당시의 시각을 기억해 둠.(파일명의 중복 방지에 활용하기 위함)
    coreNum, imgList = indexedImgList
    # 'indexedImgList'는 멀티프로세싱을 위해 총 작업량으로부터 분할받은 부분과 그 index로,
    # index 부분을 'coreNum'으로, 할당받은 자료 부분을 'imgList'로 각각 Un-packaging.
    # 'coreNum'이라고 해서 실제 CPU 내에서 해당 Core의 물리적인 할당 번호를 의미하는 것은 아니다!!
    # 중복의 방지와 진행률 표시 시의 구분을 위한 단순한 코드 상의 넘버링일 뿐이다.
    fileNames = []  # 저장한 파일의 이름을 적재할 배열을 선언.
    cnt = 0   # 저장한 파일을 계수할 변수를 0을 초기값으로 하여 선언.

    for i, img in tqdm(enumerate(imgList), total=len(imgList),
                    desc="@Core {} - Saving Crawled Images".format(coreNum)):
        # List로 가져온 Url에 대해 실제 Image를 얻어내 파일의 형태로 저장할 것이며, 총 작업량 대비 진행률을 'tqdm'을 통해 %로 표시한다.
        try:
            # Url로부터 Image를 얻어내 저장하는 과정에서, 모종의 오류가 발생하더라도 다음 Url에 대해 과정을 계속 진행할 수 있도록
            # Exception 처리를 해 둔다.
            fileName = '{}_{}_{}.jpg'.format(imgDir + timeFlag, coreNum, i)  # 지정된 형식에 따라 파일명을 정한다.
            imgSrc = urlopen(img).read()  # 'imgSrc'에 Url을 호출해 반환받은 Image를 담는다.
            buffer = open(fileName, "wb")  # 'fileName'으로 파일을 연 후 'buffer'변수에 이를 참조시킨다. Parameter "wb"는 '바이트 쓰기'를 의미한다.
            buffer.write(imgSrc)  # 열어서 'buffer'에 참조시켜 둔 파일에 'imgSrc'에 담겨있던 Image DATA의 쓰기를 실행한다.
            buffer.close()  # 쓰기 과정(저장)이 완료되면 파일을 닫는다.

            fileNames.append(fileName)  # 저장이 완료된 파일의 이름을 'fileNames'에 적재한다.
            cnt += 1  # 저장된 파일 수를 하나 증가시킨다.

        except Exception as e:
            print("\nError Occurred at " + time.strftime('%m%d - %H:%M:%S', time.localtime(time.time())))
            print("Description = {}".format(e))
            # 오류가 발생해도 이를 무시하고 프로세스는 진행되지만, 해당 오류가 발생한 시각과 내용은 출력해 두도록 하였다.
            pass

    return fileNames, cnt  # 저장한 파일들의 목록과 그 총 수를 호출측에 반환한다.


def image_crawler(indexedTagAndUrls, iteration, sleepTimeBias):
    option = webdriver.ChromeOptions()
    option.add_argument("headless")
    # 크롬 자동 실행 시, 크롬 창이 뜨며 다른 작업을 방해하는 것을 방지하기 위해 백그라운드에서 실행되도록 하는 옵션 설정 부분.

    coreNum, tagAndUrls = indexedTagAndUrls
    # 'indexedTagAndUrls'는 멀티프로세싱을 위해 총 작업량으로부터 분할받은 부분과 그 index로,
    # index 부분을 'coreNum'으로, 할당받은 자료 부분을 'tagAndUrls'로 각각 Un-packaging.
    # 'coreNum'이라고 해서 실제 CPU 내에서 해당 Core의 물리적인 할당 번호를 의미하는 것은 아니다!!
    # 중복의 방지와 진행률 표시 시의 구분을 위한 단순한 코드 상의 넘버링일 뿐이다.
    imgList = []  # 이미지 파일을 누적할 배열을 선언.

    for tag, url in tagAndUrls:  # 태그가 여러 개일 경우, 태그 수 만큼 반복 작동.
        driver = webdriver.Chrome(options=option)   # driver로 크롬을 기동.
        driver.get(url)
        time.sleep(5)  # 페이지가 온전히 Load 되도록 잠시 기다려 준다.

        page = []   # 매 스크롤 때마다의 새 페이지 정보를 담을 배열을 선언.
        tryCount = 0   # 스크롤다운 재시도 카운트를 초기화.

        for itr in tqdm(range(0, iteration), desc=("@Core {} - Scrolling ({})".format(coreNum, tag))):
            # 입력한 스크롤 반복 횟수만큼 반복문을 수행할 것이며, 전체 작업량 대비 현재 진행률을 'tqdm'을 통해 %로 표시해 줄 것이다.
            if page == BeautifulSoup(driver.page_source, 'html.parser').select('.v1Nh3.kIKUG._bz0w'):
                # 만약 스크롤 다운 이후의 page 값이 이전 내용과 같다면(스크롤 이전과 이후의 값이 같다면),
                # 스크롤 오류일 가능성이 있으므로 다시 시도하도록 할 것이다.
                if tryCount == 5:
                    print("\nScrolling is Finished Early : End of Scroll")
                    break
                    # 지정된 횟수까지 다시 시도해도 변화가 없을 경우 정말 페이지의 끝인 경우이므로, 안내 메시지를 출력한 후 스크롤을 조기 종료한다.

                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")   # 스크롤 다운을 다시 시도한다.
                time.sleep(5)  # 잠시 기다려 주도록 한다.
                tryCount += 1   # 스크롤 다운 재시도 카운트 값을 하나 증가시킨다.
                page = BeautifulSoup(driver.page_source, 'html.parser').select('.v1Nh3.kIKUG._bz0w')
                # "page"에 새로운 페이지의 'select()'구문 Parameter에 해당하는 'HTML <Class>'의 내용을 가져와 담는다.

            else:  # 스크롤 다운 수행 후의 내용이 이전 페이지의 것과 다르다면, 스크롤 다운이 정상적으로 수행됐다는 의미이다.
                tryCount = 0  # 새 page가 로드되면 스크롤 재시도 횟수를 초기화한다.
                page = BeautifulSoup(driver.page_source, 'html.parser').select('.v1Nh3.kIKUG._bz0w')
                # "page"에 새로운 페이지의 'select()'구문 Parameter에 해당하는 'HTML <Class>'의 내용을 가져와 담는다.

                try:
                    for i in page:
                        imgUrl = i.select_one('.KL4Bh').img['src']  # 가져올 Image의 Url을 얻어낸다.
                        imgList.append(imgUrl)  # 리스트에 해당 Url을 적재한다.
                except Exception as e:  # 모종의 오류가 발생하더라도, 당시까지 모은 Image DATA의 증발을 방지하기 위해, 멈추지 않고 계속 진행한다.
                    print("\nError Occurred at " + time.strftime('%m%d - %H:%M:%S', time.localtime(time.time())))
                    print("Error = {}".format(e))
                    # 프로세스를 중단하지 않고 계속 진행하되, 발생한 오류에 대해 그 시각과 내용은 출력하도록 하였다.
                    pass

                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")  # 스크롤 다운을 수행한다.
                time.sleep(3 + sleepTimeBias)  # 스크롤 다운 후, Image가 모두 온전히 Load 되도록 잠시 기다려 준다.

        driver.close()  # 측정 Tag값에 대해, 스크롤 끝까지의 Image Url을 얻어와 적재하는 과정을 마치면, 크롬을 종료시킨다.

    imgList = list(set(imgList))  # 적재된 Image Url들 간의 중복을 제거한다.(Image의 '내용'을 기준으로 중복 제거를 하는 것은 아니다.)

    return imgList   # Image들의 Url을 적재한 목록을 반환해 준다.


def insta_image_crawler_main(tagAndUrls, imgDir, iteration):
    # 'tagAndUrls'는 검색할 #Tag와 그에 대한 Url의 1:1 쌍 구성들이다. 'imgDir'은 내려받은 image를 저장할 경로이다.
    # 'iteration'은 스크롤 다운을 반복할 횟수이다.
    imgList = []  # 입력한 #Tag에 대해 스크롤다운하여 가져온 이미지들의 Url을 모아 둘 배열을 선언한다.

    while True:
        # 멀티프로세싱을 위해, 입력한 #Tag(에 대한 Url)들을 프로세스를 수행할 PC의 CPU 코어 갯수만큼씩 잘라 작업을 수행할 것이다.
        if len(tagAndUrls) <= cores:  # #Tag의 수가 Core의 수 이하라면, #Tag의 수 만큼만 작업을 분할해 수행한다.
            indexedtagAndUrls = list(enumerate(np.array_split(tagAndUrls, len(tagAndUrls))))
            sleepTimeBias = len(tagAndUrls) * 0.5
            # 스크롤 다운이 동시에 병렬로 진행되므로, 스크롤 다운 수행 후 새 페이지에 이미지가 로드되길 기다릴 시간의 기본값에 더할 연장값을 산출한다.
            for imgListPerCore in parmap.map(image_crawler, indexedtagAndUrls, iteration, sleepTimeBias):
                # 'parmap.map'으로 분할한 작업의 수에 따라 멀티프로세싱을 수행한다.
                for img in imgListPerCore:
                    imgList.append(img)  # 각 분할 작업으로부터 돌려받은 이미지의 Url 값들을 'imgList'에 더한다.
            break  # 남아있는 #Tag의 수가 코어 수 이하라는 것은 뒤에 더 남은 #Tag가 없다는 것을 의미하므로, 위의 작업이 수행되면 반복과정을 종료한다.
        else:  # 남아있는 #Tag의 수가 현재 PC의 CPU 코어 수보다 많을 경우, 다음의 과정을 진행한다.
            indexedtagAndUrls = list(enumerate(np.array_split(tagAndUrls[0:cores], cores)))
            del tagAndUrls[0:cores]
            # 'tagAndUrls'로부터 #Tag를 CPU 코어 수 만큼 가져온 다음, 전체 목록인 'tagAndUrls'에서 해당 부분은 삭제한다.
            sleepTimeBias = cores * 0.5
            # 스크롤 다운이 동시에 병렬로 진행되므로, 스크롤 다운 수행 후 새 페이지에 이미지가 로드되길 기다릴 시간의 기본값에 더할 연장값을 산출한다.
            for imgListPerCore in parmap.map(image_crawler, indexedtagAndUrls, iteration, sleepTimeBias):
                # 'parmap.map'으로 분할한 작업의 수에 따라 멀티프로세싱을 수행한다.
                for img in imgListPerCore:
                    imgList.append(img)  # 각 분할 작업으로부터 돌려받은 이미지의 Url 값들을 'imgList'에 더한다.

    indexedImgList = list(enumerate(np.array_split(list(set(imgList)), cores)))
    # 가져와 단순 추가해 둔 이미지 Url들에 대해, 단순한 Url값 기준 중복제거 처리를 수행한다.(이미지의 내용을 기준으로 비교하는 것은 아니다.)
    # 중복제거 처리 후, 멀티프로세싱을 위해 전체 양을 현재 PC의 Core의 수로 나눠 쪼개 둔다.

    fileNames = []  # 저장된 파일들의 이름을 적재할 배열을 선언한다.
    cnt = 0  # 저장된 파일들을 계수할 변수를 0으로 초기화하며 선언한다.

    for fileNamesPerCore, cntPerCore in parmap.map(image_saver, indexedImgList, imgDir):
        # 위에서 분할해 둔 Url 목록을, 파일 저장 목표 경로인 'imgDir' Parameter와 함께
        # 'parmap.map'을 통해 파일 저장 메소드에 넘겨 CPU Core 수 대로 멀티프로세싱(병렬처리)를 수행한다.
        # 각각의 프로세스로부터 반환받은 파일 이름들과 갯수에 대해 다음의 과정을 수행하게 된다.
        for fileName in fileNamesPerCore:
            fileNames.append(fileName)  # 각각의 병렬 프로세스로부터 넘겨받은 파일명 목록을 전체 목록에 더해 넣는다.
        cnt += cntPerCore  # 개별 프로세스 각각의 파일 저장 갯수를 총 갯수에 차례대로 모두 더한다.

    return fileNames, cnt   # 모든 병렬 프로세스의 파일명 목록과 갯수를 합한 것들을 호출측에 각각 반환한다.
