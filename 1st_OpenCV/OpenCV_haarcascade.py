import numpy as np
from cv2 import cv2
from pprint import pprint

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_case = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces_case:
        cv_faces = cv2.rectangle(frame, (x, y), (x + w, y + h), (188, 206, 255), 2)
        pprint(cv_faces)
        r_gray = gray[y:y + h, x:x + w]
        r_color = frame[y:y + h, x:x + w]
        eyes_case = eye_cascade.detectMultiScale(r_gray)

        for (ex, ey, ew, eh) in eyes_case:
            cv_eyes = cv2.rectangle(r_color, (ex, ey), (ex + ew, ey + eh), (109, 255, 201), 1)
            pprint(cv_eyes)

    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)

    # q 키를 누를 경우 while 반복 중지로 외부로 나감
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()