import numpy as np
from PIL import Image
import cv2
#----------------------------------------
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,90)
fontScale              = 2
fontColor              = (255,0,0)
lineType               = 7
T= False
# Blue color in BGR
color = (0, 0, 255)
thickness = 2
# Open image and make RGB and HSV versions
RGBim = Image.open("area1.jpg").convert('RGB')
HSVim = RGBim.convert('HSV')

# Make numpy versions
RGBna = np.array(RGBim)
HSVna = np.array(HSVim)
# Extract Hue
H = HSVna[:,:,0]
# Find all green pixels, i.e. where 100 < Hue < 140
lo,hi = 100,140
# Rescale to 0-255, rather than 0-360 because we are using uint8
lo = int((lo * 255) / 360)
hi = int((hi * 255) / 360)
L_green = np.where((H>lo) & (H<hi))
array_X=L_green[1]
array_Y=L_green[0]
#----------Extraire Coordinate of object-------------------------
c1 = (900.0, 360.0) # Coor Pt1
c2 = (1024.0, 420.0) # Coor Pt2
Pts = np.arange(8).reshape(4,2)
for i in range (4):
    for j in range (2):
        if i<2:
            Pts[i][j]=int(c1[j])
        else:
            Pts[i][j]=int(c2[j])
Pts[1][1]=int(c2[1])
Pts[3][1]=int(c1[1])
#----------------------------------------------------------------------
for i in range (len(array_X)):
    for j in range (4):
        if (Pts[j][0] == array_X[i]) and (Pts[j][1] == array_Y[i]):
            T= True
if T==True :
    cv2.putText(RGBna,'Warning Obstacle!!!!!!!',bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
else:
    cv2.putText(RGBna,'Safe Road', bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
resize_Res = cv2.resize(RGBna,(1080,720))
cv2.imshow("Run",resize_Res)
cv2.waitKey(0)






