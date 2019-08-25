from collections import deque
import prediction as pre
import tensorflow as tf
import _thread as thr
import pandas as pd
import numpy as np
import time
import cv2
import csv
import os
# import minpy.numpy as np
# import matplotlib.pyplot as plt

data = deque(maxlen=50)
x_axis = []
y_axis = []
z_axis = []
timestamps = []
frameNum = 0
x,y,w,h = 0, 0, 0, 0
# tempSpeed = [0.0, 0.0]
# currentSpeed = [0.0, 0.0]
# a, tempa, area = 0, 0, 0
a, area = 0, 0
tempY = 0

FLAG_FALL = False

def timestamp(convert_to_utc=False):
    t = time.time()
    return int(round(t * 1000))

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
# cap.set(cv2.CAP_PROP_FPS, 25)
# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error: opening video stream or file")


def fall_or_not(data, frameNum):
    global FLAG_FALL

    if (len(data)<50):
        pass
    elif frameNum%20 == 0:
        res = pre.prediction(data)
        if (res[0] == 0 and res[1] > 0.88):
            FLAG_FALL = True
            print('fall: {}'.format(res[1]))
        elif (res[0] == 1):
            print('lying: {}'.format(res[1]))
        elif (res[0] == 2 and res[1] > 0.75):
            FLAG_FALL = False
            print('normal: {}'.format(res[1]))
        else:
            pass


def update_rect(frame,x,y,w,h,flag):
    color = (0,255,0)
    if flag:
        color = (0,0,255)
    else:
        color = (0,255,0)
    cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)


#
# try:
#    thr.start_new_thread(update_rect, (x,y,w,h,1, ))
# except:
#    print ("Error: cannot start the thread")

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (320,240))
    frameNum += 1
    if ret == True:
        tempframe = frame.copy()
        if (frameNum == 1):
            previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
            # print("Origin")
        if (frameNum >= 2):
            currentframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
            currentframe = cv2.absdiff(currentframe, previousframe)
            currentframe = cv2.dilate(currentframe, None, iterations = 6)
            currentframe = cv2.erode(currentframe, None, iterations = 5)
            ret, threshold_frame = cv2.threshold(currentframe, 16, 255, cv2.THRESH_BINARY)
            cnts, hierarchy = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                if cv2.contourArea(c) < 1200 or cv2.contourArea(c) > 19200:
                    continue

                (x,y,w,h) = cv2.boundingRect(c) # update the rectangle
                hull = cv2.convexHull(c)
                area = cv2.contourArea(hull)

            if (w*h==0):
                continue

            # if (frameNum % 2 == 0):
            #     currentSpeed = [x+w/2, y+h/2]
            #     a = ((currentSpeed[0]-tempSpeed[0])**2 + (currentSpeed[1]-tempSpeed[1])**2)**0.5
            #     # print(a)
            #     if (a > 30):
            #         a = tempa
            #     tempSpeed = currentSpeed
            #     tempa = a

            # ax.append(frameNum)

            ys = abs(y-tempY)
            if (ys<16):
                ys = ys
            else:
                ys = 0.1
            tempY = y

            data.append([y, (w/h)*100, area/1000*ys])

            try:
                if len(data)>=50:
                    thr.start_new_thread(fall_or_not, (data, frameNum, ))
            except:
               print ("Error: cannot start the thread")

            # timestamps.append(str(timestamp())+"{0:04d}".format(frameNum))
            # cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            update_rect(frame,x,y,w,h,FLAG_FALL)
            cv2.imshow('frame', frame)
            # cv2.imshow('threshold', currentframe)
            # cv2.imshow('gauss', gauss_image)

            if cv2.waitKey(33) & 0xFF == ord('q'):
                break

        previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)

    else:
        break


cap.release()
cv2.destroyAllWindows()
