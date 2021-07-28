import cv2 as cv
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.video import FileVideoStream
from imutils.video import  VideoStream
import argparse
import imutils
import time
import dlib

cameraID = 0

#creating camera object
camera = cv.VideoCapture(cameraID)

while True:
    #getting frame from camera
    ret, frame = camera.read()

    #showing the frame on the screen
    cv.imshow('Frame', frame)

    #defining the key