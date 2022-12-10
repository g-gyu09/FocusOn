from flask import Flask, render_template, Response
import numpy as np
import cv2
import os
import time
import picamera

import RPi.GPIO as GPIO
from PCA9685 import PCA9685



cam= cv2.VideoCapture(0)  # use 0 for web camera
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height



app = Flask(__name__)

@app.route('/')

def index():

    """Video streaming home page."""

    return render_template('index.html')


def gen_frames():  # generate frame by frame from camera
    # servo position
    pwm = PCA9685()
    pwm.setPWMFreq(50)
    pwm.setRotationAngle(1, 90) #PAN    
    pwm.setRotationAngle(0, 90) #TILT
    servoHPosition = 90
    servoVPosition = 90

    # mid face
    midFaceX = 130
    midFaceY = 130

    # wide display mid
    midScreenX = (640/2)
    midScreenY = (480/2)
    midScreenWindowY = 20
    midScreenWindowX = 20


    # motor move
    stepSize = 1;
    while True:
        
        # Capture frame-by-frame

        success, img = cam.read()  # read the camera frame
        img = cv2.flip(img, 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(20,20)
        )
        for (x,y,w,h) in faces:
            print(x,y,w,h)
            id = 'Focus On'
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
        
            midFaceX = x+(w/2)
            midFaceY = y+(h/2)
        
            if(midFaceY<(midScreenY + midScreenWindowY)):
                if(servoVPosition >= 5):
                    servoVPosition -= stepSize
                    pwm.setRotationAngle(0, servoVPosition)
                
        
            elif(midFaceY>(midScreenY - midScreenWindowY)):
                if(servoVPosition <= 175):
                    servoVPosition += stepSize
                    pwm.setRotationAngle(0, servoVPosition)
                            
            if(midFaceX< (midScreenX -midScreenWindowX)):
                if(servoHPosition >= 5):
                    servoHPosition -= stepSize
                    pwm.setRotationAngle(1, servoHPosition)
        
            elif(midFaceX > (midScreenX + midScreenWindowX)):
                if(servoHPosition <= 175):
                    servoHPosition += stepSize
                    pwm.setRotationAngle(1, servoHPosition)

            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            
        ret, buffer = cv2.imencode('.jpg', img)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'

                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                   # concat frame one by one and show result

                   # concat frame one by one and show result
@app.route('/video_feed')
def video_feed():

    """Video streaming route. Put this in the src attribute of an img tag."""

    return Response(gen_frames(),

                    mimetype='multipart/x-mixed-replace; boundary=frame')


if (__name__ == '__main__'):

    app.run(host='0.0.0.0', port=8080)

