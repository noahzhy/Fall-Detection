import cv2
import numpy as np


cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FPS, 30)
# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")
frameNum = 0
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frameNum += 1
    if ret == True:
        tempframe = frame
        if(frameNum == 1):
            previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
            print("Origin")
        if(frameNum >= 2):
            currentframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
            currentframe = cv2.absdiff(currentframe, previousframe)
            # currentframe = cv2.erode(currentframe, None, iterations=1)
            currentframe = cv2.dilate(currentframe, None, iterations=3)

            ret, threshold_frame = cv2.threshold(
                currentframe, 24, 255, cv2.THRESH_BINARY)
            gauss_image = cv2.GaussianBlur(threshold_frame, (9, 9), 0)

            # Display the resulting frame
            cv2.imshow('absdiff', currentframe)
            cv2.imshow('gauss', gauss_image)

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
