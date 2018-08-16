import cv2
import os
import numpy as np


# this converts the image to grayscale and then detects faces uing the haarcascaade 
# classifier the detected faces are returned with their starting point, height and 
# width and the first face

def detectFace(img):
    greyImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(greyImg, 1.2, 5)

    if (len(faces)==0):
        return None,None

    (x, y, w, h) = faces[0]

    if not greyImg[y:y+w, x:x+h] is None:
        return greyImg[y:y+w, x:x+h], faces[0]
    else:
        return None, None



# opens the camera and takes 50 images. detect faces in each and every image and 
# groups them into an array. it is used to train the face recognizer for a certain 
# person. trained data is then saved as a <user_name>.yml file for later use in 
# authentication

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


# Used to draw a rectangle on an image according to specific dimentions
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


# Used to draw a text on an image according to specific dimentions    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


# uses the above detectFace() function to detect faces on a given image uing a given, 
# trained face recognizer. it also returns confidence it has on the detected face and 
# the returned name this confidence is later used as avariable to detect the accuracy 
# of the detection. 

def predict(image, label_text, faceRecognizer):
    face, rect = detectFace(image)

    if not face is None:
        label, confidence = faceRecognizer.predict(face)
        
        draw_rectangle(image, rect)
        draw_text(image, label_text, rect[0], rect[1])
        
        return image, confidence
    else:
        return None, None


# this function is used to register new users
def Scan():
    name = raw_input("Enter the Username: ")
    regNewUser(name)
    print "Registering the user is complete"



# this function is used to identify a peron's face using the webcam. identification 
# is done using a faceRecognizer object loaded with the previously saved 
# <user_name>.yml file. user_name is taken as an input and the file is loaded using 
# it. currently the face area is detected and marked with a square and label of the 
# name. since only one face is used, other faces might be detected with a higher 
# confidence rate.(lower is better)

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


# this function is used to identify a peron's face using 5 photos taken from the 
# web cam. identification is done using a faceRecognizer object loaded with the 
# previously saved <user_name>.yml file. user_name is taken as an input and the 
# yml file is loaded using it. the faces are detected in all 5 images and confidence 
# from all 5 images are used to calculate an average value. this average is used 
# to determine whether the detection is succesful or not. currently the threshold 
# has been selected as 35 to decide the correctness. above 35 is considered wrong 
# and below 35 as correct.

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



# this function is used to identify a peron's face using 5 photos taken from the 
# web cam. identification is done using a faceRecognizer object loaded with the 
# previously saved <user_name>.yml file. user_name is taken as an input and the 
# yml file is loaded using it. the faces are detected in all 5 images and confidence 
# from all 5 images are used to  calculate an average value. this average is used 
# to determine whether the detection is succesful or not. currently the threshold 
# has been lowered 10 to model the false negative error. Eventhough the person is 
# right, the recognizer cannot recognize to a such high accurucy. So the recognizing
# fails

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
