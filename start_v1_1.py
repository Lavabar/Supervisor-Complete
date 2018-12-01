import os
import cv2
import numpy as np
import imutils

# importing main modules
calibrate = __import__("1_calibration_v2")
bgext = __import__("2_bg_extractor_v1")
yolo = __import__("3_object_detection_v1")

# reading video
path = "/home/user/videos/271118_1100/106_(27-11-18_11\'00\'12).avi"
cap = cv2.VideoCapture(path)

# grab frame for background
if cap.isOpened():
    global bg
    ret, bg = cap.read()
else:
    raise BaseException("Capture wasnt opened!")

#bg = imutils.resize(bg, height=1080)
# calibrating camera
params = calibrate.undistortion(bg)

# image undistorion
bg = calibrate.proc(bg, params)
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

# parameters for background extraction
bgext.img_w=bg.shape[1]
bgext.img_h=bg.shape[0]
bgext.stats = np.zeros((bgext.img_h, bgext.img_w, bgext.color_size))

# start of videostream
while cap.isOpened():
    ret, img = cap.read()
    img = calibrate.proc(img, params)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = bgext.update_bg(gray, bg)

    img_objs = yolo.get_img_objs(bg, img)
    cv2.imshow('frame', img_objs)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()