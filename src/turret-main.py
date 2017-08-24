# Nerf Turret Main File
# Messing around with OpenCV

import cv2
import numpy as np

# print module versions for fun
print(cv2.__version__)
print(np.__version__)

file_path = "C:\\Users\\A84690\\Documents\\Dev\\Comp_Vis\\nerf-turret\\images\\ugly_mug.png"

img = cv2.imread(file_path, 0)
cv2.imshow('diagram', img)
k = cv2.waitKey(0) & 0xFF

if k == ord('s'):
    file_path_bw = file_path.replace('ugly_mug', 'ugly_mug_bw')
    cv2.imwrite(file_path_bw, img)

cv2.destroyAllWindows()
