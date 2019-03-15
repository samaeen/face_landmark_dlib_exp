# import the necessary packages
from imutils import face_utils
import dlib
import cv2

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

#open video camera
cap=cv2.VideoCapture(0)

while True:
	ret,image=cap.read()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)

	for (i, rect) in enumerate(rects):
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		for (x, y) in shape:
			cv2.circle(image, (x, y), 2, (255, 255, 0), -1)
		

	cv2.imshow('img',image)
	if cv2.waitKey(30)& 0xff==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()