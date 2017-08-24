# Nerf Turret Main File
# Messing around with OpenCV
# Video killed the radio star

import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

# print module versions for fun
# print(cv2.__version__)
# print(np.__version__)

# Load classifier XMLs
face_cascade = cv2.CascadeClassifier(
    'classifiers\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('classifiers\\haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while (True):
    # Capture images frame by frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        # Display the face bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 30, 220), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Display the blue bounding eye box
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (220, 30,
                                                                    30), 2)
        # Display the center X lines
        cv2.line(frame, (x, y), (x + w, y + h), (30, 220, 30), 2)
        cv2.line(frame, (x, y + h), (x + w, y), (30, 220, 30), 2)
        cv2.line(
            frame, )

    cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

# Catch and release is a sustainable fishing method.
cap.release()
cv2.destroyAllWindows()
'''
elapsed_time_ms = 1000 * (time.clock() - before)
print('comp time = ' + '%.3f' % elapsed_time_ms + ' ms')

for (x, y, w, h) in faces:
    print('x = ' + str(x) + ', y = ' + str(y) + ', w = ' + str(w) + ', h = ' +
          str(h))
    # cv2.circle(img, (int(mid_x), int(mid_y)), 5, (30, 220, 30), -1)
    print('face center: x = ' + str(int(x + w / 2)) + ', y = ' +
          str(int(y + h / 2)))

for (ex, ey, ew, eh) in eyes:
    print('ex = ' + str(x + ex) + ', ey = ' + str(y + ey) + ', ew = ' + str(ew)
          + ', eh = ' + str(eh))

# cv2.imshow('face', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
'''
