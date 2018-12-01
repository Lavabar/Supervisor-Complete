import cv2
import numpy as np
from math import atan, isnan

# updating chosen points on image
def update_points(img, refPt, strength=0.1, zoom=1.0):
    dest_img = np.zeros(img.shape, dtype=np.uint8)

    imageHeight, imageWidth, _ = img.shape
    
    halfWidth = imageWidth / 2
    halfHeight = imageHeight / 2
    
    correctionRadius = (imageWidth**2 + imageHeight**2)**0.5 / strength

    res_refPt = np.zeros(refPt.shape)

    for y in range(imageHeight):
        for x in range(imageWidth):
            newX = x - halfWidth
            newY = y - halfHeight

            distance = (newX**2 + newY**2)**0.5
            r = distance / correctionRadius
            
            theta = 0.0
            if r == 0:
                theta = 1
            else:
                theta = atan(r) / r

            sourceX = int(halfWidth + theta * newX * zoom)
            sourceY = int(halfHeight + theta * newY * zoom)
            
            dest_img[y, x] = img[sourceY, sourceX]
            #print("x, y: %d, %d\nsourceX, sourceY: %d, %d" % (x, y, sourceX, sourceY))
            # TODO optimize
            for i in range(len(refPt)):
                if (refPt[i] == [sourceX, sourceY]).all():
                    res_refPt[i] = [y, x]
            
    return dest_img, res_refPt

# just transform image
def proc(img, params):
    strength = params[0]
    zoom = params[1]

    if strength == 0.0:
        return img
    dest_img = np.zeros(img.shape, dtype=np.uint8)

    imageHeight, imageWidth, _ = img.shape
    
    halfWidth = imageWidth / 2
    halfHeight = imageHeight / 2
    
    correctionRadius = (imageWidth**2 + imageHeight**2)**0.5 / strength

    for y in range(imageHeight):
        for x in range(imageWidth):
            newX = x - halfWidth
            newY = y - halfHeight

            distance = (newX**2 + newY**2)**0.5
            r = distance / correctionRadius
            
            theta = 0.0
            if r == 0:
                theta = 1
            else:
                theta = atan(r) / r

            sourceX = int(halfWidth + theta * newX * zoom)
            sourceY = int(halfHeight + theta * newY * zoom)
            
            dest_img[y, x] = img[sourceY, sourceX]
            
    return dest_img

# making list of lines [[p1, p2, p3, p4],...]
def make_lines_bypoints(refPt):
    refPt = np.reshape(refPt, (4, 4, 2))
    lines = []
    lines.extend(refPt)
    for i in range(4):
        line = []
        for j in range(4):
            line.append(refPt[j, i])
        lines.append(line)

    lines = np.asarray(lines)
    #print(lines)
    return lines

# function for calcualting distance between point and line
def distance(line):
    p1 = line[0]
    p2 = line[1]
    p3 = line[2]
    p4 = line[3]

    d1 = abs((p4[1] - p1[1]) * p2[0] - (p4[0] - p1[0]) * p2[1] + p4[0] * p1[1] - p4[1] * p1[0]) /\
         ((p4[1] - p1[1]) ** 2 + (p4[0] - p1[0]) ** 2) ** 0.5

    d2 = abs((p4[1] - p1[1]) * p3[0] - (p4[0] - p1[0]) * p3[1] + p4[0] * p1[1] - p4[1] * p1[0]) /\
         ((p4[1] - p1[1]) ** 2 + (p4[0] - p1[0]) ** 2) ** 0.5

    return d1, d2

# function for calculating standard deviation
def standard_dev(lines):
    ds = []
    for line in lines:
        d1, d2 = distance(line)
        #print("d1, d2 = %f, %f" % (d1, d2))
        ds.append(d1)
        ds.append(d2)

    sum_ds = sum(ds)
    mo_ds = sum_ds / len(ds)
    ds = np.asarray(ds)
    st_dev = (np.sum(np.square(np.subtract(ds, mo_ds))) / ds.shape[0]) ** 0.5
    if isnan(st_dev):
        return 1.0 
    return st_dev


def undistortion(image):
    # list of points
    refPt = []
    # helps to find out where to make next point 
    st_model = "\n - - - \n - - - \n - - - \n - - - \n"
    # get coordinates of mouse cursor after clicking
    def get_coords(event, x, y, flags, param):
        # grab references to the global variables
        nonlocal refPt, image, st_model

        if event == cv2.EVENT_LBUTTONDOWN:
            refPt.append([x, y])
            cv2.rectangle(copy_image, (x-2, y-2), (x + 2, y + 2), (0, 0, 255), 3)
            # lighting next point
            st_model = st_model.replace(" ", "*", 1)
            print(st_model)
    
    copy_image = np.copy(image)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", get_coords)
    # beginning process of choosing points on image
    # lighting first point
    st_model = st_model.replace(" ", "*", 1)
    print(st_model)
    while True:
        cv2.imshow("image", copy_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if len(refPt) != 16:
        raise BaseException("Wrong number of points(should be 16)")

    refPt = np.asarray(refPt)

    # begining iterations
    lines = make_lines_bypoints(refPt)
    st_dev = standard_dev(lines)
    print("st_dev:")
    print(st_dev)
    if st_dev <= 1.0:
        return (0.0, 1.0)
    strengths = [i/30 for i in range(1, 18, 2)]
    for strength in strengths:
        new_image, refPt = update_points(image, refPt, strength=strength, zoom=1.0)
        refPt = np.int32(refPt)
        image = new_image
        lines = make_lines_bypoints(refPt)
        st_dev = standard_dev(lines)
        print(st_dev)
        if st_dev <= 1.0:
            return (strength, 1.0)

    return (0.0, 1.0)

if __name__ == "__main__":
    path = "test_imgs/undist_test.jpg"
    img = cv2.imread(path)
    strength = undistortion(img)
    new_image = proc(img, strength, 1.0)
    cv2.imshow("undistort", new_image)
    key = False
    while not key:
            key = cv2.waitKey(1) & 0xFF == ord('q')
    cv2.destroyAllWindows()