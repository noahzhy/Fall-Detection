import cv2
import numpy as np
cap = cv2.VideoCapture(0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
fgbg = cv2.createBackgroundSubtractorMOG2()
# fourcc = cv2.VideoWriter_fourcc(*"XVID")
frame1 = np.zeros((640, 480))
 # out = cv2.VideoWriter("test424.avi", fourcc, 10, (640, 480))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("1", frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    fgmask = fgbg.apply(frame)
    mask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    cv2.imshow("2", mask)
    k = cv2.waitKey(33) & 0xff
    if k == 27:
        break
    contours, _ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if 100 < cv2.contourArea(c) < 40000:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
            # out.write(frame)
# out.release()
cap.release()
cv2.destoryAllWindows()
