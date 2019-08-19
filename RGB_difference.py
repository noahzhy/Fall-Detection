import cv2
import numpy as np


cap = cv2.VideoCapture('video_02.avi')
# cap.set(cv2.CAP_PROP_FPS, 30)
# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

frameNum = 0
x,y,w,h = 0, 0, 0, 0
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frameNum += 1
    if ret == True:
        tempframe = frame.copy()
        if(frameNum == 1):
            previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
            print("Origin")
        if(frameNum >= 2):
            currentframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
            currentframe = cv2.absdiff(currentframe, previousframe)
            currentframe = cv2.dilate(currentframe, None, iterations = 7)
            currentframe = cv2.erode(currentframe, None, iterations = 6)

            # currentframe = cv2.morphologyEx(currentframe, cv2.MORPH_GRADIENT, kernel)

            ret, threshold_frame = cv2.threshold(currentframe, 16, 255, cv2.THRESH_BINARY)
            gauss_image = cv2.GaussianBlur(threshold_frame, (7, 7), 0)
            cnts, hierarchy = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                if cv2.contourArea(c) < 2000:
                    continue
                # c是一个二值图，boundingRect是矩形边框函数，用一个最小的矩形，把找到的形状包起来；
                # x,y是矩形左上点的坐标；w,h是矩阵的宽和高
                (x,y,w,h) = cv2.boundingRect(c)
                # rectangle画出矩形，frame是原图，(x,y)是矩阵的左上点坐标，(x+w,y+h)是矩阵右下点坐标
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            # Display the resulting frame
            cv2.imshow('frame', frame)
            cv2.imshow('gauss', gauss_image)
            cv2.imshow('threshold', currentframe)

            # Press Q on keyboard to  exit
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break
        previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
