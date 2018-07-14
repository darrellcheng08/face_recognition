import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

cam = cv2.VideoCapture(0)
fontFace = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf < 50):    
            if(Id==1):
                name = "Darrell Cheng"
            elif(Id==2):
                name = "Ashley Tumapon"
        else:
            name="Unknown"

        cv2.putText(im, str(name), (x,y+h), fontFace, 1.0, (255, 255, 255), 1)
    cv2.imshow('FACE RECOGNITION', im) 
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()