def get_fake_face(saveDir):
    # 함수 호출 시, 얻어 온 Image를 파일로 저장할 경로를 'saveDir' Parameter로 넘겨받아 온다.
    import urllib.request
    import time
    import cv2
    import os

    timeFlag = time.strftime('%Y%m%d', time.localtime(time.time()))
    # 모듈 최초 호출 시의 연,월,일 정보를 기억해 둔다.(파일명의 중복 방지에 활용하기 위함)
    url = 'https://www.thispersondoesnotexist.com/image'  # nVidia의 Fake Face Image를 얻을 수 있는 Url이다.
    user_agent = "Chrome/86.0.4240.75"  # 본인이 사용할 브라우저의 정보를 입력해 넣는다.(버전 등)
    request = urllib.request.Request(url, headers={'User-Agent': user_agent})
    # 'urllib.request'모델의 'Request'클래스를 주어진 Parameter 값으로 객체를 생성해 'request' 변수에 대입해 둔다.

    i = 0  # 가져온 Image의 일련번호 할당에 활용할 변수를 0으로 초기화하며 선언한다.

    while True:
        # 사용자가 강제로 프로세스를 중단하기 전까지 무한정 작동할 것이다.
        try:
            timeNow = time.strftime('%Y%m%d', time.localtime(time.time()))
            # 실시간으로 연,월,일을 기억해 둔다.(파일명의 중복 방지에 활용하기 위함)
            if int(timeFlag) < int(timeNow):
                # 이전의 연월일 값(기준값)과 현재의 연월일 값을 비교해 차이가 발생할 경우(=날짜가 달라진 경우)를 탐지한다.
                timeFlag = timeNow  # 날짜의 변경이 감지됐다면, 기준값을 새로 변경된 값으로 갈아넣어 준다.
                i = 0  # 일련번호 부여의 기준인 날짜가 변경됐으므로, 일련번호를 초기화한다.

            html = urllib.request.urlopen(request).read()
            # 'request' 변수에 참조시켜 둔 Class 객체의 내용을 토대로 Url에 접속해 해당 Url Page의 HTML문 내용(Image)을 얻어온다.
            fileName = '{}{}_{}.jpg'.format(saveDir, timeFlag, i)
            # Parameter로 넘겨받은 경로와 현재 연/월/일, 일 중 일련번호를 조합해 파일명을 조합한다.
            buffer = open(fileName, 'wb')
            # 'wb'='바이트 쓰기'와 위에서 조합한 파일명('fileName')을 Parameter로 하여 파일을 생성해 열고 임시 변수에 이를 참조시켜 둔다.
            buffer.write(html)
            # 앞서 미리 받아 둔 Image를 열어 둔 파일에 기록한다.
            buffer.close()
            # 파일로 기록해 저장한 뒤엔 해당 파일을 닫아 종료시킨다.

            # 아래의 과정은 'jpg'로 받아와 그대로 저장한 Image를 'png'형식으로 다시 저장하고 원본 .jpg 파일을 삭제하는 것이다.
            # 이러한 과정을 추가로 거치는 이유는 Machine/Deep learning에는 png 형식을 활용하는 편이 보다 효율적이기 때문이다.
            file = cv2.imread(fileName)  # openCV Library의 'imread' 메소드를 이용해 저장한 앞서 jpg 파일을 다시 불러온다.
            pngName = '{}{}_{}.png'.format(saveDir, timeFlag, i)
            # 앞서 .jpg 파일로 저장할 때와 확장자를 제외하곤 동일한 형식으로 파일명을 조합한다.
            cv2.imwrite(pngName, file)  # openCV Library의 'imwrite' 메소드를 이용해 앞서 조합한 파일명으로 Image를 다시 저장한다.
            os.remove(fileName)  # 남아있는 원본 jpg 파일은 삭제한다.

            i += 1  # 일련번호를 1 증가시킨다.

        except Exception as e:
            # 만약 네트워크 오류 등으로 이미지를 가져오지 못한 때가 생기더라도, 프로세스가 종료되지 않고 계속 수행되도록
            # Exception 처리를 하였다.
            # 프로세스가 중지되지 않는 대신 오류가 발생한 시각과 그 내용은 출력해 두도록 하였다.
            print("Error Occurred at " + time.strftime('%m%d - %H:%M:%S', time.localtime(time.time())))
            print("Description = {}".format(e))
