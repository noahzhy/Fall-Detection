import cv2
# import kalman
# import numpy as np
# import minpy.numpy as np
import matplotlib.pyplot as plt

# draw the picture
ax = []                    # 定义一个 x 轴的空列表用来接收动态的数据
ay = []                    # 定义一个 y 轴的空列表用来接收动态的数据
rate_w_h = []
rate_speed = []
area_w_h = []
plt.ion()                  # 开启一个画图的窗口

# plt.clf()
# count = 0



cap = cv2.VideoCapture('video_02.avi')
# cap.set(cv2.CAP_PROP_FPS, 30)
# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

frameNum = 0
x,y,w,h = 0, 0, 0, 0
tempSpeed = [0.0, 0.0]
currentSpeed = [0.0, 0.0]
a = 0
tempa = 0
area = 0

# Read until video is completed
while (cap.isOpened()):
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

            # currentframe = cv2.morphologyEx(currentframe, cv2.MORPH_GRADIENT, kernel)

            ret, threshold_frame = cv2.threshold(currentframe, 16, 255, cv2.THRESH_BINARY)
            # gauss_image = cv2.GaussianBlur(threshold_frame, (7, 7), 0)
            cnts, hierarchy = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                if cv2.contourArea(c) < 1200:
                    continue
                # c是一个二值图，boundingRect是矩形边框函数，用一个最小的矩形，把找到的形状包起来；
                # x,y是矩形左上点的坐标；w,h是矩阵的宽和高
                (x,y,w,h) = cv2.boundingRect(c) # update the rectangle
                hull = cv2.convexHull(c)
                # cv2.drawContours(frame, [hull], 0, (0, 0, 255), 2)
                area = cv2.contourArea(hull)

            if (w*h==0):
                continue

            if (frameNum % 2 == 0):
                currentSpeed = [float(x+w/2), float(y+h/2)]
                a = ((currentSpeed[0]-tempSpeed[0])**2 + (currentSpeed[1]-tempSpeed[1])**2)**0.5
                # print(a)
                if (a > 30):
                    a = tempa
                tempSpeed = currentSpeed
                tempa = a

            rate_speed.append(a* 3)

            plt.clf()
            plt.title('Fall Analysis')
            ax.append(frameNum)
            ay.append(y+h/2)                  # y of mid-point
            rate_w_h.append(float(w/h)*100)     # rate of w/h
            area_w_h.append(area/100)

            plt.plot(ax, ay, color='blue', label='y of mid-point')
            plt.plot(ax, rate_w_h, color='green', label='rate of w/h')
            plt.plot(ax, rate_speed, color='red', label='rate of speed')
            plt.plot(ax, area_w_h, color='orange', label='area of motion')
            plt.legend()                        # 显示图例
            plt.xlabel('Frames')


            # plt.draw()                        # for ubuntu
            plt.pause(0.01)                     # for windows


            # rectangle画出矩形，frame是原图，(x,y)是矩阵的左上点坐标，(x+w,y+h)是矩阵右下点坐标
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            # Display the resulting frame
            cv2.imshow('frame', frame)
            cv2.imshow('threshold', currentframe)
            # cv2.imshow('gauss', gauss_image)

            # Press Q on keyboard to  exit
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break
        previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
    # Break the loop
    else:
        break


plt.pause(5)
plt.ioff()
# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
