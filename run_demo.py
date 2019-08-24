import os
import cv2
import csv
import time
import pandas as pd
import numpy as np
import tensorflow as tf
# import minpy.numpy as np
# import matplotlib.pyplot as plt


# draw the picture
ax = []                    # 定义一个 x 轴的空列表用来接收动态的数据
ay = []                    # 定义一个 y 轴的空列表用来接收动态的数据
rate_w_h = []
rate_speed = []
area_motion = []
timestamps = []
# plt.ion()                  # 开启一个画图的窗口

def timestamp(convert_to_utc=False):
    t = time.time()
    return int(round(t * 1000))

# plt.clf()
# count = 0
cap = cv2.VideoCapture(0)
# cap.set(3, 320) #设置分辨率
# cap.set(4, 240)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# cap.set(cv2.CAP_PROP_FPS, 30)
# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

frameNum = 0
x,y,w,h = 0, 0, 0, 0
tempSpeed = [0.0, 0.0]
currentSpeed = [0.0, 0.0]
a, tempa, area = 0, 0, 0
tempY = 0
# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if (cap.get())
    # frame = cv2.resize(frame, (320,240))
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

            if (frameNum % 2 == 0):
                currentSpeed = [x+w/2, y+h/2]
                a = ((currentSpeed[0]-tempSpeed[0])**2 + (currentSpeed[1]-tempSpeed[1])**2)**0.5
                # print(a)
                if (a > 30):
                    a = tempa
                tempSpeed = currentSpeed
                tempa = a

            # ax.append(frameNum)
            ay.append(y)                    # y of mid-point
            ys = abs(y-tempY)
            if (ys<15):
                ys = ys
            else:
                ys = 0.1
            tempY = y
            area_motion.append(area/1000*ys)
            rate_w_h.append((w/h)*100)     # rate of w/h

            timestamps.append(str(timestamp())+"{0:04d}".format(frameNum))

            # plt.plot(ax, ay, color='blue', label='X: y of mid-point')
            # plt.plot(ax, rate_w_h, color='green', label='Y: w/h')
            # plt.plot(ax, area_motion, color='orange', label='Z: area of motion')

            # plt.legend()                        # 显示图例
            # plt.xlabel('Frames')
            #
            # # plt.draw()                        # for ubuntu
            # plt.pause(0.01)                     # for windows

            # rectangle画出矩形，frame是原图，(x,y)是矩阵的左上点坐标，(x+w,y+h)是矩阵右下点坐标
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            # Display the resulting frame
            cv2.imshow('frame', frame)
            # cv2.imshow('threshold', currentframe)
            # cv2.imshow('gauss', gauss_image)

            # Press Q on keyboard to  exit
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break
        previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
    # Break the loop
    else:
        break

# data = {
#     # 'action': 'n/a',
#     # 'timestamp': timestamps,
#     'X': ay,
#     'Y': rate_w_h,
#     'Z': area_motion  # speed of motion area
# }
#
# df = pd.DataFrame(data)

# fileName = path.split('.')[0].split('/')[-1]
# df.to_csv("dataset/{}.csv".format(fileName), mode='w',index=False, header=['action', 'timestamp', 'x-axis', 'y-axis', 'z-axis'])
# print(df)
# kalman.show_data(df)

# plt.pause(5)
# plt.ioff()
# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()