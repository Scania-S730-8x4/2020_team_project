def file_renamer(srcDir, dstDir):
    # 'srcDir'은 작업 대상 파일을 불러올 경로, 'dstDir'은 작업 완료 파일을 다시 저장할 경로이며, 이 둘은 동일할 수 있다.
    from tqdm import tqdm
    import os

    files = os.listdir(srcDir)  # 작업 대상 파일 경로 내에 존재하는 파일들의 이름 목록을 변수에 대입시켜 둔다.
    s_num, cnt = 0, 0  # 작업 대상 파일의 일련번호와 작업을 진행한 횟수를 셀 변수들을 초기값을 0으로 해 선언한다.

    for file in tqdm(files):  # 작업을 진행하며, 총 작업량 대비 진행률을 'tqdm'을 통해 %로 표시한다.
        newName = 'sorted_0' + str("{0:06d}".format(s_num)) + '.png'  # 대상 파일들의 새 이름 형식을 일련번호를 활용해 지정한다.
        srcName = srcDir + file   # 원본 대상 파일의 경로와 파일명을 서로 결합한다.
        dstName = dstDir + newName   # 작업 완료 파일의 경로와 새 파일명을 서로 결합한다.

        try:
            os.rename(srcName, dstName)  # 'os' Library의 'rename'메소드를 통해 'srcName'인 파일을 'dstName'로 재명명한다.
            s_num += 1
            cnt += 1   # 작업에 성공하면, 일련번호와 작업 성공 횟수를 각각 1씩 증가시킨다.
        except FileExistsError as e:   # 만약 파일 이름이 중복되어 오류가 발생한 경우에 거칠 과정이다.
            flag = 0   # 우선 flag 변수를 0을 초기값으로 해 선언한다. 이는 while문 반복 여부 확인용 지표로 활용될 것이다.
            while flag == 0:  # 특정 과정이 수행돼 flag의 값이 0이 아닌 다른 값으로 변경되기 전까지 아래의 과정을 계속 반복할 것이다.
                s_num += 1  # 일련번호만을 1 증가시킨다.
                newName = 'sorted_0' + str("{0:06d}".format(s_num)) + '.png'  # 증가시킨 일련번호로 새 파일명을 구성한다.
                dstName = dstDir + newName  # 작업 완료 파일 경로와 새 파일명을 결합한다.
                try:
                    os.rename(srcName, dstName)  # 새로 구성한 파일명으로 재명명을 다시 시도해 본다.

                    cnt += 1  # 오류가 다시 발생하지 않고 성공적으로 파일이 저장될 경우, 작업 성공 수를 1 증가시킨다.
                    flag = 1  # while문을 빠져나갈 수 있도록 'flag(지표)'를 0이 아닌 다른 값으로 변경시켜 준다.
                except FileExistsError as e:
                    continue
                    # 만약 다시금 오류가 발생한 경우(= 증가시킨 일련번호 파일명도 이미 존재할 경우),
                    # 일련번호를 다시 증가시켜 재명명을 시도하는 과정을, 오류가 발생하지 않을 때까지(이미 존재하지 않는 일련번호 값에 도달할 때까지) 반복하도록 한다.

    print("Total Renamed Count : {} files".format(cnt))  # 작업에 성공한 횟수를 출력해 안내하고, Module의 프로세스를 종료한다.
