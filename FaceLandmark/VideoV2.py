import cv2
import dlib


detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
video_capture = cv2.VideoCapture(0) #Webcam object

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #clahe_image = clahe.apply(gray)

    detections = detector(gray, 1) #Detect the faces in the image

    for (k,d) in enumerate(detections):

        shape = predictor(gray, d)
        for i in range(1,68):
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2)
    cv2.imshow("image", frame)
    if cv2.waitKey(30)& 0xff==ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()