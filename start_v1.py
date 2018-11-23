import os
import cv2
import numpy as np

# importing main modules
calibrate = __import__("1_calibration_v1")
bgext = __import__("2_bg_extractor_v1")
yolo = __import__("3_object_detection_v1")

# listing all frames
path = "test_videos/Crowd_PETS09/S1/L1/Time_13-59/View_001/"
ls = os.listdir(path)
ls.sort()

# grab frame for background
bg = cv2.imread(path+ls[0])
# calibrating camera
strength = calibrate.undistortion(bg)
zoom = 1.0
# image undistorion
bg = calibrate.proc(bg, strength, zoom)
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

# parameters for background extraction
bgext.img_w=bg.shape[1]
bgext.img_h=bg.shape[0]
bgext.stats = np.zeros((bgext.img_h, bgext.img_w, bgext.color_size))

# start of videostream
for fr_name in ls:
    img = cv2.imread(path+fr_name)
    img = calibrate.proc(img, strength, zoom)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = bgext.update_bg(gray, bg)

    img_objs = yolo.get_img_objs(bg, img)
    cv2.imshow('frame', img_objs)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()