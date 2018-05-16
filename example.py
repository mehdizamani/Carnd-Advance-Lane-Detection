import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
%matplotlib qt

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')
test_image= cv2.imread('camera_cal/calibration3.jpg')
def find_chessboard_points(images_directory):
    tmp_objpoints = [] # 3d points in real world space
    tmp_imgpoints = [] # 2d points in image plane.
    # Step through the list and search for chessboard corners
    for fname in images_directory:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        image_size= gray.shape[::-1]
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            tmp_objpoints.append(objp)
            tmp_imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
           # cv2.imshow('img',img)
           # cv2.waitKey(500)
    print(len(tmp_objpoints))  # only in 17 photos our of 20 photos, the cv2.findChessboardCorners() could correctly find the corners
    cv2.destroyAllWindows()
    return tmp_objpoints,tmp_imgpoints

#Write a function that takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)    
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imshow('undist',undist)
    return undist

print("EEEE")
objpoints,imgpoints = find_chessboard_points(images)

#undistorted = cal_undistort(test_image, objpoints, imgpoints)
