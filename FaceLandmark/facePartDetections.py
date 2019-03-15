import numpy as np
import cv2
import dlib
from imutils import face_utils
import imutils

d="shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(d)



cap=cv2.VideoCapture(0)
while True:
	ret,image=cap.read()
	gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)

	for (i, rect) in enumerate(rects):
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			clone = image.copy()
			cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
			for (x, y) in shape[i:j]:
				cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
				(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
				roi = image[y:y + h, x:x + w]
				roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
			#cv2.imshow("ROI", roi)
			#cv2.imshow("Image", clone)

		output = face_utils.visualize_facial_landmarks(image, shape)
		cv2.imshow("Image", output)
	if cv2.waitKey(30)& 0xff==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
