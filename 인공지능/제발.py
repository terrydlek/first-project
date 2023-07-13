import cv2
import numpy as np
from djitellopy import Tello
import time

drone = Tello()  # 텔로 선언
drone.connect()  # 텔로 연결
print(drone.get_battery())  # 잔여 배터리 출력
drone.streamon()  # 카메라 켜기  # 이륙
drone.takeoff()
drone.send_rc_control(0, 0, 23, 0)
time.sleep(2.2)

width, height = 1440, 1280  # openCV 윈도우 크기 180, 120에 센터 점을 위치 시킬 예정
fbRange = [6200, 6800]  # 얼굴 인식 사각형 면적의 적절한 범위, 해당 범위 밖이면 앞뒤 움직여 조절
lrPid = [0.4, 0.4, 0]  # 드론 회전 속도 보정용 계수
udPid = [0.4, 0.4, 0]  # 드론 상하 속도 보정용 계수
pLrError = 0  # 가로 180와 틀어진 정도, 틀어진 만큼 회전을 통해 보정
pUdError = 0  # 세로 120과 틀어진 정도, 틀어진 만큼 위아래 움직여 보정


def findFace(img):  # img에서 얼굴을 찾아 리턴
    faceCascade = cv2.CascadeClassifier("C:/Users/USER/Downloads/haarcascade_frontalface_default.xml")  # 사전학습된 얼굴 인식 파일 임포트
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 얼굴 인식을 위한 흑백화 이미지
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)  # 얼굴 인식

    myFaceListC = []  # 얼굴 중심점 저장하는 list
    myFaceListArea = []  # 얼굴 면적 저장하는 list

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255),
                      2)  # 얼굴 인식하고 사각형 그리기, 시작점 (x, y)와 끝점 (x + w, y + h)를 빨간선 두께 2로 그리기
        cx = x + w // 2  # 센터 x는 x와 w의 반을 더해 구함
        cy = y + h // 2  # 센터 y는 y와 h의 반을 더해 구함
        area = w * h  # 얼굴 면적은 w와 h를 곱해 구함
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)  # 얼굴 중심점에 지름이 5인 녹색 원을 속을 채워서 그림
        myFaceListC.append([cx, cy])  # 얼굴 중심점 list에 추가
        myFaceListArea.append(area)  # 얼굴 면적 list에 추가

    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]


def trackFace(info, w, h, lrPidLocal, udPidLocal, pLrErrorLocal, pUdErrorLocal):
    area = info[1]  # findFace에서 구한 얼굴 면적
    x, y = info[0]  # findFace에서 구한 얼굴 중심점
    fb = 0  # 드론 앞뒤로 움직일 속도

    lrError = x - w // 2  # 이미지 중심과 얼굴 중심점의 좌우 오차
    udError = h // 2 - y  # 이미지 중심과 얼굴 중심점의 상하 오차
    lrSpeed = lrPidLocal[0] * lrError + lrPidLocal[1] * (lrError - pLrErrorLocal)  # 드론 좌우 보정 속도
    udSpeed = udPidLocal[0] * udError - udPidLocal[1] * (pUdErrorLocal - udError)  # 드론 상하 보정 속도
    lrSpeed = int(np.clip(lrSpeed, -50, 50))  # 드론 속도 최소 최대 제한
    udSpeed = int(np.clip(udSpeed, -100, 100))  # 드론 속도 최소 최대 제한

    if area > fbRange[0] and area < fbRange[1]:  # 드론과 얼굴이 적정거리일 때
        fb = 0  # 호버링
    elif area > fbRange[1]:  # 드론과 얼굴이 가까울 때
        fb = 20  # 후진
    elif area < fbRange[0] and area != 0:  # 드론과 얼굴이 멀 때
        fb = -20  # 전진

    if x == 0:  # 인식되지 않았을 때
        lrSpeed = 0
        udSpeed = 0
        error = 0

    drone.send_rc_control(0, fb, 0, lrSpeed)
    return error


while True:
    img = drone.get_frame_read().frame  # 드론에서 카메라 프레임 가져오기
    img = cv2.resize(img, (width, height))  # 카메라 프레임 리사이즈
    img, info = findFace(img)  # 얼굴 인식하고, 중심점, 면적 가져오기
    pLrError, pUdError = trackFace(info, width, height, lrPid, udPid, pLrError, pUdError)  # 얼굴 추적하여 움직이기
    cv2.imshow("Output", img)  # Output이라는 이름의 윈도우 열고 처리된 이미지 출력
    if cv2.waitKey(1) & 0xFF == ord('q'):  # q키 누르면
        drone.land()  # 착륙
        break  # while 나가기

cv2.destroyWindow("Output")  # Output이라는 이름의 창 닫기
drone.end()  # 드론 연결 종료
