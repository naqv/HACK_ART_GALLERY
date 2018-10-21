import requests
import json
import cv2

#########################################################
# AUTHOR : S.P. MOHANTY
#########################################################
addr = 'http://iccluster026.iccluster.epfl.ch:5001'
test_url = addr + '/api/style/rain_princess'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('scenery.jpg')
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
print(img_encoded)
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
# decode response
# print(response.text) # JPEG binary content ! Write to a file as "name.jpg"

fp=open("test.jpg", "wb")
fp.write(response.content)
fp.close()