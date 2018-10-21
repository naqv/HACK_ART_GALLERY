#!/usr/bin/python
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw
import imutils

#dictionary of all contours
contours = {}
#array of edges of polygon
approx = []
#scale of the text
scale = 2
#camera
cap = cv2.VideoCapture(0)
print("press q to exit")
url = "http://10.4.230.168:8080/shot.jpg"

_width  = 600.0
_height = 420.0
_margin = 0.0

count = 0

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

corners = np.array(
	[
		[[  		_margin, _margin 			]],
		[[ 			_margin, _height + _margin  ]],
		[[ _width + _margin, _height + _margin  ]],
		[[ _width + _margin, _margin 			]],
	]
)

pts_dst = np.array( corners, np.float32 )

#calculate angle
def angle(pt1,pt2,pt0):
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]
    return float((dx1*dx2 + dy1*dy2))/math.sqrt(float((dx1*dx1 + dy1*dy1))*(dx2*dx2 + dy2*dy2) + 1e-10)

def filter_1(frame):
    return cv2.bitwise_not(frame)

def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))
    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1], pts[1][0]:pts[2][0]]
                       
    return img_crop

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(frame,80,240,3)
        #contours
        canny2, contours, hierarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for i in range(0,len(contours)):
            approx = cv2.approxPolyDP(contours[i],cv2.arcLength(contours[i],True)*0.02,True)

            if(abs(cv2.contourArea(contours[i]))<100 or not(cv2.isContourConvex(approx))):
                continue

            if(len(approx)==4):
                vtc = len(approx)
                cos = []
                for j in range(2,vtc+1):
                    cos.append(angle(approx[j%vtc],approx[j-2],approx[j-1]))
                cos.sort()
                mincos = cos[0]
                maxcos = cos[-1]
                x,y,w,h = cv2.boundingRect(contours[i])
                print (w,h)
                
                if(vtc==4):
                    if (w > 100 or h >100):
                        pts_src = np.array(approx, np.float32 )
                        h, status = cv2.findHomography( pts_src, pts_dst )
                        # cv2.drawContours(frame, [approx], -1, ( 255, 0, 0 ), 24)
                        rect = cv2.minAreaRect(contours[i])
                        img_croped = crop_minAreaRect(frame, rect)
                        # filter_1(img_croped)
                        # cv2.imshow("crop_img", filter_1(img_croped))  # aqui estamos cortando solo el pedazo de papel que necesitamos
                        cv2.drawContours(frame, [approx], -1, (255, 0, 0), 24)
                        cv2.imwrite("framewd.jpg", img_croped)
                        with open("framewd.jpg", "rb") as imageFile:
                            f = img_croped.read()
                            b = bytearray(f)
                            print(b)
                        delay_x = delay_y = 50

                        # frame[delay_x:delay_x+ img_croped.shape[0], delay_y:delay_y+img_croped.shape[1]] = filter_1(img_croped)

        #Display the resulting frame
        out.write(frame)
        cv2.imshow('frame',frame)
        cv2.imshow('canny',canny)
        if cv2.waitKey(1) == 1048689: #if q is pressed
            break

#When everything done, release the capture
cap.release()
cv2.destroyAllWindows()