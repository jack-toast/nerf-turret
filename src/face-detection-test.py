# Nerf Turret Main File
# Messing around with OpenCV

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

file_path = 'images\\ugly_mug.png'

img = cv2.imread(file_path, 1)
height, width, channels = img.shape
print('height = ' + str(height) + " px\nwidth = " + str(width) + ' px')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

before = time.clock()
faces = face_cascade.detectMultiScale(gray, 1.2, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

elapsed_time_ms = 1000 * (time.clock() - before)
print('comp time = ' + '%.3f' % elapsed_time_ms + ' ms')
cv2.imshow('face', img)
cv2.waitKey(5000)
cv2.destroyAllWindows()
