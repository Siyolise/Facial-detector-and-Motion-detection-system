import cv2
import numpy as np

#Fetch pre-trained classifiers for face stored as xml files.
#create a face_cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Create video capture object
cap = cv2.VideoCapture(0)

while True:
    ret,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces =  face_cascade.detectMultiScale(gray,1.1,4)
    #draw a rectangle face box
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y), (x+w, y+h),(0,0,255),2)
            
    cv2.imshow("img",img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
