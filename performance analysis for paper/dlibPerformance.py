from imutils import face_utils
import dlib
import cv2

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

img=cv2.imread('angled.jpg',cv2.IMREAD_COLOR)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
rects = detector(gray, 0)

for (i, rect) in enumerate(rects):
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	for (x, y) in shape:
		cv2.circle(img, (x, y), 2, (255, 255, 0), -1)

cv2.putText(img, "Looking left", (25, 50),cv2.FONT_HERSHEY_SIMPLEX, .7, (66, 134, 244), 2)

cv2.imwrite("dlib_perform1.jpg",img)
cv2.imshow('img',img)
cv2.waitKey()