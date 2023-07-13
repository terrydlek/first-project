from djitellopy import tello    # tello를 사용
import numpy as np
import cv2    # 영상처리하는 라이브러리 opencv
import time

me = tello.Tello()    # me에 tello 넣어주기
me.connect()    # 연결
me.takeoff()
me.send_rc_control(0, 0, 15, 0)
me.streamon()

time.sleep(2.2)

w, h = 500, 400    # 화면의 가로, 세로
fbRange = [6200, 6800]    # 전진과 후진범위
pid = [0.4, 0.4, 0]    # 회전반경
pError = 0



def findFace(img):
    faceCascade = cv2.CascadeClassifier("C:/Users/USER/Downloads/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)
    myFaceListC = []    # cx, cy, me위치
    myFaceListArea = []    # 면적 값

    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x + w, y + h), (0, 0, 255), 2)   # (시작점, 끝점, 색상, 두께)
        cx = x + w // 2    # 중앙
        cy = y + h // 2    # 중앙
        area = w * h    # 면적은 너비 * 높이
        cv2.circle(img, (cx,cy), 5, (0,255,0), cv2.FILLED)
        # (이미지, 원의 중심, 원의 반지름, 색상, 선의 타입)
        myFaceListC.append([cx,cy])    # 리스트요소에 추가
        myFaceListArea.append(area)
    if len(myFaceListArea) != 0:    # 면적에 아무것도 없으면
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]    # cx, cy, area == 0


def trackFace(info, w, pid, pError):
    area = info[1]
    x, y = info[0]
    fb = 0

    error = x - w//2    # 이미지의 중심
    speed = pid[0] * error + pid[1] * (error - pError)    # pid 제어법
    speed = int(np.clip(speed, -50, 50))    # numpy을 활용해 스피드 정의

    # fbRange[0]보다 크고 fbRange[1]보다 작으면 정비, 녹색영역
    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:    # 가까이 있어서 뒤로 물러남.
        fb = 20
    elif area < fbRange[0] and area != 0:    # 멀어서 가까이 다가감.
        fb = -20

    if x == 0:    # 그냥 0이면 모두 0으로 바꿔버림
        speed = 0
        error = 0
    me.send_rc_control(0, fb, 0, speed)
    return error



while True:
    img = me.get_frame_read().frame    # 프레임을 가져와서 나에게 전달
    img = cv2.resize(img, (1280, 1440))    # 이미지 크기 조절
    img, info = findFace(img)
    pError = trackFace(info, w, pid, pError)
    cv2.imshow("Image", img)    # 이미지 출력
    if cv2.waitKey(1) & 0xFF == ord("q"):# 동영상 지연
        me.land()
        break