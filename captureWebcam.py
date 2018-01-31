# capture.py
import numpy as np
import cv2

# Capture video from camera
cap = cv2.VideoCapture(0)
count = 1

while(True):
#    print('frame number: %d' % count)
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (640, 360))
#    gray = gray[50:300, 200:400]
#    cv2.imwrite("/Users/alejandro/AnacondaProjects/cv/No_hands/frame_NH_%d.jpg" % count, gray)
    count += 1

    # Display the resulting frame
    cv2.imshow('frame', gray)
    
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
exit()