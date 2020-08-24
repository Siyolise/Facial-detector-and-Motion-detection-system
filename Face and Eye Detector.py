import cv2
import numpy as np

#Fetch pre-trained classifiers for face and eyes stored as xml files.
#create face and eye detector with openCV
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

#Load image input in color mode.
img = cv2.imread("Syoh.jpg")
#resizing the image
rs_img = cv2.resize(img,(300,400))
#converting image into grayscale mode
gray = cv2.cvtColor(rs_img, cv2.COLOR_BGR2GRAY)

#Find face in the image.
faces =  face_cascade.detectMultiScale(gray,1.2,5)
'''For faces found they must have rectangular face box using rs_img,co-ordinates
,rgb values for rectangle outline and a width of the rect.''' 
for (x,y,w,h) in faces:
    cv2.rectangle(rs_img,(x,y), (x+w, y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = rs_img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
            
cv2.imshow("img",rs_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
