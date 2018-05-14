import cv2
import os
import numpy as np

def detectFace(img):
    greyImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(greyImg, scaleFactor = 1.2, minNeighbors = 5)

    if (len(faces)==0):
        return None,None

    (x, y, w, h) = faces[0]

    if not greyImg[y:y+w, x:x+h] is None:
        return greyImg[y:y+w, x:x+h], faces[0]
    else:
        return None, None

def regNewUser(name):
    labels = []
    faces = []
    fileName = name + ".yml"
    
    capture = cv2.VideoCapture(0)

    for i in range(1,51):
        ret, image = capture.read()
        if ret:
            face, rect = detectFace(image)

            if face is not None:
                faces.append(face)
                labels.append(1)
    else:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.train(faces, np.array(labels))
        face_recognizer.save(fileName)     


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(image, label_text, faceRecognizer):
    face, rect = detectFace(image)

    if not face is None:
        label, confidence = faceRecognizer.predict(face)
        
        draw_rectangle(image, rect)
        draw_text(image, label_text, rect[0], rect[1])
        
        return image, confidence
    else:
        return None, None

def Scan():
    name = raw_input("Enter the Username: ")
    regNewUser(name)
    print "Registering the user is complete"

def Identify():
    name = raw_input("Enter the Username: ")
    fileName = name + ".yml"
    
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(fileName)

    capture = cv2.VideoCapture(0)

    while(True):
        ret, image = capture.read()
        if not image is None:
            if ret:
                predictedImage, percentage = predict(image, name, face_recognizer)
                
                if not predictedImage is None:
                    cv2.imshow("Identifying..", predictedImage)
                    cv2.waitKey(10)

def check():
    name = raw_input("Enter the Username: ")
    fileName = name + ".yml"
    
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(fileName)

    capture = cv2.VideoCapture(0)

    totalconfidence = 0
    count = 0
    for i in range(0,5): 
        ret, image = capture.read()
        if not image is None:
            if ret:
                predictedImage, confidence = predict(image,name, face_recognizer)
                if not confidence is None:
                    totalconfidence += confidence
                    count += 1
    else:
        confidence = totalconfidence/count
        if (confidence<35):
            print "It is ", name
        else:
            print "Cannot Identify"

def checkWithFalseNegtive():
    name = raw_input("Enter the Username: ")
    fileName = name + ".yml"
    
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(fileName)

    capture = cv2.VideoCapture(0)

    totalconfidence = 0
    count = 0
    for i in range(0,5): 
        ret, image = capture.read()
        if not image is None:
            if ret:
                predictedImage, confidence = predict(image,name, face_recognizer)
                if not confidence is None:
                    totalconfidence += confidence
                    count += 1
    else:
        confidence = totalconfidence/count
        if (confidence<10):
            print "It is ", name
        else:
            print "Cannot Identify"



choice = raw_input("Type \"Login\" or, \"Register\": ")

if (choice=="Register"):
    Scan()
elif (choice=="Login"):
    #Identify()
    #check()
    checkWithFalseNegtive()
else:
    print "chose correct option"



