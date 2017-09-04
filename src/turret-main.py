# Nerf Turret Main File
# Messing around with OpenCV
# Video killed the radio star

import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

# Load classifier XMLs
face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('classifiers/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
width = cap.get(3)
height = cap.get(4)

print('width = ' + str(width) + ', height = ' + str(height))


def add_visuals(frame):
    # Display the center X lines

    cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 30, 220), 2)
    cv2.line(frame, (x, y), (x + w, y + h), (30, 220, 30), 2)
    cv2.line(frame, (x, y + h), (x + w, y), (30, 220, 30), 2)
    cv2.line(frame, (int(width / 2), int(height / 2)), (xmid, ymid), (30, 220,
                                                                      30), 2)


while (True):
    # Capture images frame by frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Display the face bounding box
        xmid = int(x + w / 2)
        ymid = int(y + h / 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # Display the blue bounding eye box
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (220, 30,
                                                                    30), 2)

        # Add the boxes and lines on the frame.
        add_visuals(frame)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # TODO Put the x and y offsets here
        xoffset = int(xmid - width / 2)
        yoffset = int(height / 2 - ymid)
        offsets = str(xoffset) + ', ' + str(yoffset)
        cv2.putText(frame, offsets, (0, int(height * 0.06)), font, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

# Catch and release is a sustainable fishing method.
cap.release()
cv2.destroyAllWindows()
