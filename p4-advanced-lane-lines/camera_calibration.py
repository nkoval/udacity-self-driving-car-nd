import numpy as np
import cv2
import glob
import pickle

# Load images for calibration
images = glob.glob('./camera_cal/calibration*.jpg')

# Define the board shape
board_shape = (9, 6)

objpoints = []
impoints = []

objp = np.zeros((board_shape[0] * board_shape[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:board_shape[0], 0:board_shape[1]].T.reshape(-1, 2)

for idx, image in enumerate(images):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imshape = gray.shape

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, board_shape, None)

    if ret:
        objpoints.append(objp)
        impoints.append(corners)

# Calculate camera's distortion matrices
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, impoints, imshape[::-1], None, None)

data_pickle = {}
data_pickle["mtx"] = mtx
data_pickle["dist"] = dist

# Save the data to use it later
pickle.dump(data_pickle, open("./callibration_pickle", "wb"))