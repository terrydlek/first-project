import time

from djitellopy import tello
import cv2

drone = tello.Tello()
drone.connect()
print(drone.get_battery())

drone.streamon()

while True:
    img = drone.get_frame_read().frame
    img = cv2.resize(img, (144, 96))
    cv2.imshow("Image", img)
    cv2.waitKey(1)

    drone.takeoff()
    time.sleep(3)

    drone.up(50)
    time.sleep(3)
