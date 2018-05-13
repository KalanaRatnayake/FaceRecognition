import cv2
import os
import numpy as np


def prepareUsers():
    users = ["", ]
    fileObj = open("users.txt", "r")

    for name in fileObj:
        users.append(name)
    else:
        fileObj.close()
        return users

def regUser(name):
    fileObj = open("users.txt", "a")
    fileObj.write(name+"\n")
    fileObj.close()

def regNewUser(name):
    tempUsers = prepareUsers()
    
    dirPath = "Dataset/s"+str(len(tempUsers))
    os.mkdir(dirPath)
    
    capture = cv2.VideoCapture(0)

    for i in range(1,31):
        img_path = dirPath + "/"+str(i)+".jpg"
        
        ret, image = capture.read()
        if ret:
            cv2.imwrite(img_path, image)
            cv2.imshow("Scanning...", image)
            cv2.waitKey(100)
    else:
        regUser(name)


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
        return None

def prepareTrainingData():
    dirList = os.listdir("Dataset")
    
    faces = []
    labels = []
    
    for dir_name in dirList:       
        if not dir_name.startswith("s"):
            continue;

        label = int(dir_name.replace("s", ""))
    
        subject_dir_path = "Dataset/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue;
            
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            
            face, rect = detectFace(image)
            
            if face is not None:
                faces.append(face)
                labels.append(label)
    
    return faces, labels

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(img, face_recognizer):
    users = prepareUsers();
    
    face, rect = detectFace(img)

    if not face is None:
        label, confidence = face_recognizer.predict(face)

        label_text = users[label]
        
        draw_rectangle(img, rect)
        draw_text(img, label_text, rect[0], rect[1])
        
        return img, confidence, label_text
    else:
        return None, None, None

def Scan():
    name = raw_input("Enter the name: ")
    regNewUser(name)
    print "Registering the user is complete"

def Identify():
    faces, labels = prepareTrainingData()
    
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))

    capture = cv2.VideoCapture(0)

    while(True):
        ret, image = capture.read()
        if not image is None:
            if ret:
                predictedImage, percentage, label_text = predict(image, face_recognizer)
                if not predictedImage is None:
                    cv2.imshow("Identifying", predictedImage)
                    cv2.waitKey(10)

def check():
    faces, labels = prepareTrainingData()
    
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))

    capture = cv2.VideoCapture(0)

    ret, image = capture.read()

    if not image is None:
        if ret:
            predictedImage, percentage, label_text = predict(image, face_recognizer)
            if (percentage>50):
                print "It is ", label_text
                cv2.imshow("Identifying", predictedImage)
                cv2.waitKey(10000)

            else:
                print "Cannot Identify"
                cv2.imshow("Cannot Identify", predictedImage)
                cv2.waitKey(10000)


choice = raw_input("Type \"Identify\" to Identify a person, \"Scan\" to add a person to Databse: ")

if (choice=="Scan"):
    Scan()
    Identify()
elif (choice=="Identify"):
    #Identify()
    check()
else:
    print "chose correct option"



