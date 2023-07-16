import requests
import sys
import os
import datetime
import win32api
import time
from ast import literal_eval


def checkContinue(dir):
    ###########################################################################################
    now = datetime.datetime.now()
    now = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute)
    result = "result_" + now + ".txt"
    # print(now)
    ###########################################################################################
    # 여러 디렉토리를 점검할 수 있도록 인자로 받도록 설정
    if len(sys.argv) == 1:
        print("디렉토리 옵션을 최소 1개이상 주셔야 합니다. \
        \n\nex) python 5.malwareCom_hashCheck_loop.py \"C:\\Download\" \"C:\\Download2\" \
        ")
    else:
        api_key_file = "C:/Temp/api_key.txt"
        if os.path.isfile(api_key_file):
            # print("API Key 파일이 존재합니다.")
            data = open(api_key_file, "r", encoding="euckr")
            data = data.read().strip()
            api_key = data.split('\n', 2)[0]
            # print(vtkey)
        else:
            print("API Key 파일이 존재하지 않습니다.")
            api_key = input("멀웨어닷컴에서 받은 API 키를 입력해 주세요 : ")
            inputKey = input(
                "입력한 API키를 디스크(c:\\Temp\\api_key.txt)에 저장하시겠습니까?\n(저장하시면 다음 실행시부터 API Key를 입력하실 필요가 없습니다.)(Y or N): ")
            if inputKey == 'Y' or inputKey == 'y':
                print("API Key를 생성합니다.")
                with open(api_key_file, "w") as f:
                    f.write(str(api_key))

    j = 1
    while len(sys.argv) > 1:
        print(len(sys.argv))

        print("#########################       %d회차 점검 시작       ###########################" % (j))
        print("#################################################################################")
        if len(sys.argv) > 1 and api_key:
            print("ok")
            imsiDir = "imsi/"
            if not (os.path.isdir(imsiDir)):  # imsi 디렉토리가 없으면 만든다.
                os.makedirs(os.path.join(imsiDir))

            # try:
            for i in range(1, len(sys.argv)):
                print("#################################################################################")
                print("=================================================================================")
                print(" 프로그램이 다운로드한 파일에 대하여 실시간 악성코드 감염여부를 검사합니다.")
                print("=================================================================================")
                print("#################################################################################")
                filenames = os.listdir(sys.argv[i])
                print(filenames)
                print("================================================================")
                print("%d번째 점검 디렉토리 : \t'%s'" % (i, sys.argv[i]))
                print("================================================================")
                print("점검 파일목록 : \t", filenames)
                print("================================================================")

                for (path, dirs, filenames) in os.walk(sys.argv[i]):
                    for filename in filenames:
                        full_filename = os.path.join(path, filename)
                        print(full_filename)
                        os.system('certUtil -hashfile "' + full_filename + '" MD5 > ' + imsiDir + 'imsi_hash.txt')
                        txtf = open(imsiDir + "imsi_hash.txt", "r", encoding="euckr")
                        data = txtf.read()
                        second_line = data.split('\n', 2)[1]
                        hash = second_line.replace(" ", "")
                        print("파일명: ", full_filename, "###########\t\t md5 해쉬: ", hash)


                        url = f"https://public.api.malwares.com/api/v22/file/analysis/{hash}"
                        headers = {
                            "x-api-key": api_key,
                        }

                        response = requests.request("GET", url, headers=headers)
                        json_response = response.json()
                        # json_response = str(json_response).replace("\'", "\"")
                        print(json_response)


                        if json_response['result_code'] == int(200):  # or json_response['result_code'] == int(2):
                            pass
                            print("서버에 해시정보가 있습니다. 보내주신 해시값과 서버의 해시값을 비교합니다.")
                            f = open(imsiDir + hash + ".txt", "w")
                            f.write(str(json_response))
                            # f.write(json_response)
                            f.close()

                            with open(imsiDir + hash + ".txt") as f:
                                # data2 = json.load(json_file)
                                # data2 = eval(f.read())
                                data2 = literal_eval(f.read())  # str -> dic로 읽는다.

                                ai_score = data2['data']['ai_score']
                                print(ai_score)
                                if 'signcheck' in data2 and data2['signcheck']:
                                    signcheck = data2['signcheck']['verified']
                                else:
                                    signcheck = 'None'
                                print(signcheck)
                                check = "파일이름 : " + str(filename) + "\nai_score : " + str(
                                    ai_score) + "\nsigncheck : " + str(signcheck) + "\n\nDetection Malware!!!!!! "
                                if signcheck == "unsigned" or ai_score > int(3):
                                    # if signcheck == 'None' or signcheck == "unsigned" or ai_score > int(3):
                                    win32api.MessageBox(None, check, 'Warning!!!!', 0)
                            print("----------------------------------------------------------------")

                        elif json_response['result_code'] == int(2):
                            print(full_filename, " 이 파일은 현재 분석 진행중입니다. 5분 후에 다시 시도 하시기 바랍니다.")
                            print("----------------------------------------------------------------")

                        elif json_response['result_code'] == int(404):
                            print(full_filename, " 이 파일은 서버에 해시정보가 없습니다. 악성파일 가능성이 있으므로 직접 홈페이지에 파일업로드하여 점검하시기 바랍니다.")

                            check = "파일이름 : " + str(
                                filename) + "\n\n이 파일은 서버에 해시정보가 없습니다.\n악성파일 가능성이 있으므로 직접 아래 홈페이지에 파일업로드하여 점검하시기 바랍니다.\n\nMalware Suspicion!!!!!!\n\nhttps://www.malwares.com/ "
                            win32api.MessageBox(None, check, 'Warning!!!!', 0)
                            print("----------------------------------------------------------------")

                        else:
                            # json_response['result_code'] == int(0)
                            print("요청시 잘못된 파라미터, 사용권한 없음 및 API키 오류 등이 발생하였습니다. 정상적으로 다시 요청하여 주시기 바랍니다.")
                            print("----------------------------------------------------------------")

                        txtf.close()
                    time.sleep(1)
            # except:
            #     pass
            #     print("폴더명 또는 API Key 값이 이상합니다. 다시 확인해 주시기 바랍니다.")

        else:
            print("\nAPI Key를 정확히 입력하셔야 점검이 시작됩니다.\n")
        ###########################################################################################
        j = j + 1
        time.sleep(10)


if __name__ == "__main__":
    # Code in this block will run when stepper.py is invoked from the command line
    checkContinue(dir=sys.argv)
