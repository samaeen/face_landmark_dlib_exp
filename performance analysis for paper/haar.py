import cv2
import numpy as np

face_cascade=cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

eye_cascade=cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')


img=cv2.imread('angled.jpg',cv2.IMREAD_COLOR)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=3,minSize=(80, 80),flags=0)

for (x,y,w,h) in faces:
	cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	roi_gray=gray[y:y+h,x:x+w]
	roi_color=img[y:y+h,x:x+w]

cv2.putText(img, "Face not detected", (25, 50),cv2.FONT_HERSHEY_SIMPLEX, .7, (66, 134, 244), 2)

cv2.imwrite("haar_perform1.jpg",img)
cv2.imshow('img',img)
cv2.waitKey()