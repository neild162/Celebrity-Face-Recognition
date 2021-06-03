import os
import cv2 as cv
import numpy as np

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']     #list of all the people recognizer will have
DIR = r'C:\Users\neil1\OneDrive\Desktop\Neil\OpenCV\train'   #path of the folder containing the 5 folders having the images

haar_cascade = cv.CascadeClassifier('haar_face.xml')  #our haarcascade classifier

features = []  #training set will consist of 2 lists
labels = []  #for every face in the features list, what is its corresponding label. Whose face is it i.e. our label
 
def create_train():  #will loop over all the 5 folders and all the images in the 5 folders, get the faces and add all those images to training set
    for person in people:   #loop over every person in 'people' list
        path = os.path.join(DIR, person)    #grabbing path of each individual folder of the 5 people
        label = people.index(person)        #label is basically index of 'people' list.ex img of Elton John will have label=1

        for img in os.listdir(path):  #now we are inside each individual folder and we now loop through all the images in each folder
            img_path = os.path.join(path,img)   #grabbing image path 

            img_array = cv.imread(img_path)    #now that we have the path for the image(s), we are going to read in the image(s)
            if img_array is None:
                continue 
                
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)   #converting to gray-scale

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)   #image DETECTION

            for (x,y,w,h) in faces_rect:    #loop over every face in faces_rect
                faces_roi = gray[y:y+h, x:x+w]   #faces_regionofinterest. basically cropping out everything else in img and keeping only the face
                features.append(faces_roi)   #appending the faces to 'features' list created above
                labels.append(label)

create_train()
print('Training done ---------------')

features = np.array(features, dtype='object')  #converting to numpy arrays
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()    #instantiating face recognizer

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)