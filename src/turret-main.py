# Nerf Turret Main File
# Messing around with OpenCV

import cv2
import numpy as np
from matplotlib import pyplot as plt

# print module versions for fun
print(cv2.__version__)
print(np.__version__)

file_path = "C:\\Users\\A84690\\Documents\\Dev\\Comp_Vis\\nerf-turret\\images\\ugly_mug.png"

img = cv2.imread(file_path, 0)
"""
cv2.imshow('diagram', img)
k = cv2.waitKey(0)

if k == ord('s'):
    print('saving ugly_mug_bw.png')
    cv2.imwrite(file_path.replace('ugly_mug', 'ugly_mug_bw'), img)

cv2.destroyAllWindows()
"""
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.show()
