import cv2
import numpy as np
from scipy.ndimage import median_filter

nframes=100
img_w=640
img_h=480
color_size=256
n = 0

def update_bg(img, bg):
    global stats, n
    for y in range(img_h):
        for x in range(img_w):
            stats[y, x, img[y, x]] += 1

    n += 1
    if n == nframes:
        bg = np.argmax(stats, axis=2)
        stats = np.zeros((img_h, img_w, color_size))
        n = 0
    
    return median_filter(np.array(bg, np.uint8), 3)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    ret, bg = cap.read()
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        bg = update_bg(gray, bg)

        cv2.imshow("bg", bg)
        cv2.imshow("frame", gray)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
