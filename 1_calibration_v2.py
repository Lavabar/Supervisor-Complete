import numpy as np
import cv2

def undistortion(image):
    img_gr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    square_size = 2.0

    pattern_size = (6, 4)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    image_points = []
    
    h, w = image.shape[:2]
    refPt = []
    # helps to find out where to make next point 
    st_model = "\n - - - - - \n - - - - - \n - - - - - \n - - - - - \n"
    # get coordinates of mouse cursor after clicking
    def get_coords(event, x, y, flags, param):
        # grab references to the global variables
        nonlocal refPt, img_gr, st_model

        if event == cv2.EVENT_LBUTTONDOWN:
            refPt.append([x, y])
            cv2.rectangle(copy_image, (x-2, y-2), (x + 2, y + 2), 255, 3)
            # lighting next point
            st_model = st_model.replace(" ", "*", 1)
            print(st_model)
    
    copy_image = np.copy(img_gr)
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

    if len(refPt) != 24:
        raise BaseException("Wrong number of points(should be 24)")

    refPt = np.asarray(refPt, dtype=np.float32)
    refPt = np.expand_dims(refPt, axis=1)
    print(refPt)
    image_points.append(refPt.reshape(-1, 2))
    obj_points.append(pattern_points)

    print('ok')

    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, image_points, (w, h), None, None)

    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())

    cv2.destroyAllWindows()

    return (camera_matrix, dist_coefs)


def proc(img, params):
    camera_matrix = params[0]
    dist_coefs = params[1] 
    size = (img.shape[1], img.shape[0])
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (size[0], size[1]), 1, (size[0], size[1]))
    
    if not all(roi):
        raise BaseException("calibration was failed. please, restart!")

    x, y, w, h = roi

    new_img = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
    new_img = new_img[y:y+h, x:x+w]
    #res = np.empty_like(new_img)
    #res = new_img[:]
    res = new_img.copy()
    return res
    #return new_img