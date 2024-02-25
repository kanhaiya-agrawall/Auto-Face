import numpy as np
import cv2 as cv
import os
import pandas as pd
from datetime import date
from datetime import datetime
from csv import writer

def distance(x1, x2):

    #Eucledian Distance
    return np.sqrt(((x1-x2)**2).sum())

def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        ix = train[i,:-1]
        iy = train[i, -1]

        #Computing distance from the test point
        d = distance(test, ix)
        dist.append([d, iy])

        #Sorting  distance and getting top k of them
    dk = sorted(dist, key=lambda x: x[0])[:k]

    #Retrieving the labels
    labels = np.array(dk)[:, -1]
    #Frequencies of each label
    output = np.unique(labels, return_counts=True)
    #Maximum frequency and label corresponding to that frequency
    index = np.argmax(output[1])
    return output[0][index]

#Camera Initialization
cam_capture = cv.VideoCapture(0)

#Detection Of Face
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

#Training data (x)
face_data = []

data_location = './Face Data/'

#Label (y)
label = []

class_id = 0 #0 to n

#Dictionary for mapping (ID<->Name)
names_dict = {}

# Data preparation

#Loading the training data
for fx in os.listdir(data_location):
    if fx.endswith('.npy'):

        #Mapping between class_id and name
        names_dict[class_id] = fx[:-4]
        data_item = np.load(data_location+fx)
        face_data.append(data_item)

        #Labels
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        label.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(label, axis=0).reshape((-1, 1))

#Single Matrix
trainset = np.concatenate((face_dataset, face_labels), axis=1)

face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

#Testing
while True:
    ret, frame = cam_capture.read()

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.2, 10)

    for face in faces:
        x, y, w, h = face

        #ROI
        offset = 10
        facial_region = frame[y-offset: y+h+offset, x-offset:x+w+offset]
        facial_region = cv.resize(facial_region, (100, 100))
        out = knn(trainset, facial_region.flatten())

        #Predicted Name
        pred_name = names_dict[int(out)]

        #Display the name and rectangle around it
        cv.putText(frame, pred_name, (x, y-10),
                   cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0),2,cv.LINE_AA)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    cv.imshow("Faces", frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('x'):
        break

print(pred_name)
today = date.today()
date_time = datetime.now()
print("Datetime from timestamp:", date_time)

list = [pred_name, date_time]

with open('attendance_report '+str(today)+'.csv', 'a') as f_object:
    writer_object = writer(f_object)
    writer_object.writerow(list)
    f_object.close()
    

#Releasing all devices
cam_capture.release()
cv.destroyAllWindows()