#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 23:40:36 2018

@author: natachaquintero
"""

import requests
import cv2
import numpy as np
import imutils


url = "http://10.4.230.168:8080/shot.jpg"

while True:
    new_image= requests.get(url)
    img_arr= np.array(bytearray(new_image.content),dtype=np.uint8)
   
    # convert the frame to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 150)

	# find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    img=cv2.imdecode(img_arr,-1)
    cv2.imshow("natacha",img)
    if cv2.waitKey(1)==27:
        break
