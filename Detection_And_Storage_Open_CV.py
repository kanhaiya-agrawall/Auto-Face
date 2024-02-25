import cv2 as cv
import numpy as np

#Camera Initialization
cam_capture = cv.VideoCapture(0)

#Detect Face
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

append_face=0
face_data=[]
data_location='./Face Data/'
file_name=input("Enter Name: ")
#Loop breaks when x is pressed
while True:
    #Reading frames
    bool_return, frame = cam_capture.read()
    #To save space we will convert frame to grayscale
    gray_color = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #If read incorrectly, skip
    if bool_return == False:
        continue

    #List Of Faces (Rectangles), Scaling Factor, Number of neighbors
    faces = face_cascade.detectMultiScale(gray_color, 1.2, 10)
    #Largest (Face Reverse Sorting)
    faces= sorted(faces,key=lambda f:f[2]*f[3],reverse=True)

    # cv.imshow('Video Frame',frame)
    # cv.imshow('Video Frame gray',gray_color)

    #Rectangle around face,vertices,color,thickness
    for(x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)

        #Extract Region Of Interest
        offset=10
        facial_region=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        facial_region=cv.resize(facial_region,(100,100))

        #Every tenth face
        if append_face%10==0:
            face_data.append(facial_region)

    cv.imshow('Video Frame', frame)

#Last 8 bits
    key_pressed = cv.waitKey(1) & 0xFF
    if key_pressed == ord('x'):
        break

#Converting data into numpy array
face_data=np.asarray(face_data)

#Reshaping to 1-D
face_data=face_data.reshape((face_data.shape[0],-1))

#Saving
np.save(data_location+file_name+'.npy',face_data)

#Releasing all devices
cam_capture.release()
cv.destroyAllWindows()