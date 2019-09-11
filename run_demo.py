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


data = deque(maxlen=50)
x_axis = []
y_axis = []
z_axis = []
warning_timestamp = 0
frameNum = 0
x,y,w,h = 0, 0, 0, 0
a, area = 0, 0
motion_speed = 0
tempY = 0
fall_cancel = 0.4
fall_count = 0

FLAG_FALL = False
FLAG_WARNING = False

video_path = 'E:/Fall-Detection/cam5.avi'
# video_path = 0
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 25.0, (320,240))

def timestamp(convert_to_utc=False):
    t = time.time()
    return int(t)

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 25)

if (cap.isOpened() == False):
    print("Error: opening video stream or file")

def fall_or_not(data, frameNum):
    global FLAG_FALL
    global FLAG_WARNING
    global fall_cancel
    global fall_count
    global warning_timestamp

    if (frameNum%15 == 0):
        # if FLAG_WARNING:
        #     warning_status()
        if x == 0 or x+w == 320:
            pass
        else:

            res = pre.prediction(data)
            if (res[0] == 0 and res[1] > 0.80):
                FLAG_FALL = True
                fall_count += 1
                # fall_cancel = 0.4
                # if (FLAG_WARNING == False):
                #     warning_timestamp = timestamp()
                #     FLAG_WARNING = True
                # else:

                print('fall: {}'.format(res[1]))
            elif (res[0] == 1):
                print('lying: {}'.format(res[1]))
            elif (res[0] == 2 and res[1] > fall_cancel):
                FLAG_FALL = False
                fall_count = 0
                # FLAG_WARNING = False
                # fall_cancel = 0.4
                print('normal: {}'.format(res[1]))
            elif (res[0] == 3):
                pass
            # print('sleep: {}'.format(res[1]))


# def warning_status():
#     global FLAG_FALL
#     global fall_cancel
#     # if FLAG_WARNING:
#     if (timestamp() - warning_timestamp >= 2):
#         FLAG_FALL = True
#         fall_cancel = 0.73


def update_rect(frame,x,y,w,h,flag_fall):
    # default is green
    color = (0,255,0)
    text = 'normal'
    # if flag_warning:
    if flag_fall and fall_count > 2 :
        color = (0,0,255)
        text = 'fall'
        img2 = cv2.imread('top.png')
        img2 = cv2.resize(img2,(320, 240))
        frame = cv2.addWeighted(frame,0.7,img2,0.3,0)

    frame = cv2.putText(frame, 'Status: {}'.format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    frame = cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
    return frame


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
            # currentframe = cv2.GaussianBlur(currentframe, (7, 7), 0)
            currentframe = cv2.absdiff(currentframe, previousframe)
            currentframe = cv2.dilate(currentframe, None, iterations = 7)
            currentframe = cv2.erode(currentframe, None, iterations = 5)
            ret, threshold_frame = cv2.threshold(currentframe, 20, 255, cv2.THRESH_BINARY)
            cnts, hierarchy = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                if cv2.contourArea(c) < 1300 or cv2.contourArea(c) > 19200:
                    continue

                (x,y,w,h) = cv2.boundingRect(c) # update the rectangle
                hull = cv2.convexHull(c)
                area = cv2.contourArea(hull)

            if (w*h==0):
                continue

            if (y-tempY)<-10:
                if (fall_count > 0):
                    fall_count -= 1

            ys = y-tempY
            if (ys<16):
                ys = ys
            else:
                ys = 1
            tempY = y

            motion_speed = area/500*ys
            if (abs(motion_speed) < 120):
                pass
            else:
                motion_speed = 0

            data.append([y-30, (w/h-0.5)*100, motion_speed])

            if len(data)>=50:
                try:
                    thr.start_new_thread(fall_or_not, (data, frameNum, ))
                except:
                    print ("Error: cannot start the thread")

            # timestamps.append(str(timestamp())+"{0:04d}".format(frameNum))
            # cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            cv2.imshow('frame', update_rect(frame,x,y,w,h,FLAG_FALL))
            # out.write(frame)
            # cv2.imshow('threshold', currentframe)
            # cv2.imshow('gauss', gauss_image)

            if cv2.waitKey(33) & 0xFF == ord('q'):
                break

        previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)

    else:
        break

cap.release()
# out.release()
cv2.destroyAllWindows()
