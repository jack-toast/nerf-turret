# Nerf Turret Main File
# Messing around with OpenCV

import cv2
import numpy as np

# print module versions for fun
print(cv2.__version__)
print(np.__version__)

img = cv2.imread(
    "C:\\Users\\A84690\\Documents\\IOT project\\Diagrams\\ENV_wiring_version_1_bb.png",
    0)
cv2.imshow('diagram', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
