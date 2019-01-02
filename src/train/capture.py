import cv2
import os
import time
import subprocess
import sys
import datetime

cv2.namedWindow("preview")
vc = cv2.VideoCapture(1)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    now = datetime.datetime.now()
    cv2.imshow("preview", frame)
    cv2.imwrite('./tmp'+str(sys.argv[1])+'/'+str(now.month)+str(now.day)+str(now.hour)+str(now.minute)+str(now.second)+str(now.microsecond)+"webcam.jpg" , frame) # SIDE EFFECT: frame.jpg file
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")


