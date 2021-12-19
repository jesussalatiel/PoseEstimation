import cv2
import numpy as np
import time

cap = cv2.VideoCapture("curl.mp4")

while True:

    #success, img = cap.read()
    #mg = cv2.resize(img, (720, 360))
    

    img = cv2.imread("curl.jpg")
    cv2.imshow("Trainner", img)
    cv2.waitKey(1)
