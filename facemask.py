import os
import cv2
import random
import argparse
import numpy as np
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA

def confidence(array):
    x = 0
    if len(array) > 10:
        for i in array[-10:]:
            if i == 0.0:
                x += 1
    return (x/10, "{0}%".format((x/10)*100))

parser = argparse.ArgumentParser()
parser.add_argument('--train', help='Train model', type=eval, choices=[True, False], default='False')
args = parser.parse_args()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# User message
font = cv2.FONT_HERSHEY_SIMPLEX

# Read video
capture = cv2.VideoCapture(0)

if args.train:
    data = []

    # Loop through dir and process without mask images
    path = os.path.dirname(os.path.realpath(__file__)) + '/without_mask/'
    for image in os.listdir(path):
        image = path+os.path.basename(image)
        image = cv2.imread(image, 0)
        faces = face_cascade.detectMultiScale(image)
        if len(faces) != 0:
            for (x, y, w, h) in faces:
                face = image[y:y+h, x:x+w]
                face = cv2.resize(face, (50, 50))
                data.append(face)

    np.save('without_mask.npy', data)
    print("Saved without mask numpy data...")

if args.train:
    data = []

    # Loop through dir and process masked images
    path = os.path.dirname(os.path.realpath(__file__)) + '/with_mask/'
    for image in os.listdir(path):
        image = path+os.path.basename(image)
        image = cv2.imread(image, 0)
        faces = face_cascade.detectMultiScale(image)
        if len(faces) != 0:
            for (x, y, w, h) in faces:
                face = image[y:y+h, x:x+w]
                face = cv2.resize(face, (50, 50))
                data.append(face)

    np.save('with_mask.npy', data)
    print("Saved with mask numpy data...")

if not args.train:
    # Load npy face data if available
    try:
        np_no_mask = np.load('without_mask.npy')
        np_with_mask = np.load('with_mask.npy')
    except IOError:
        print("No npy data found. Please run program again with --train argument...")
        exit(-1)
    # Reshape face data
    np_no_mask = np_no_mask.reshape(np_no_mask.shape[0], 50*50)
    np_with_mask = np_with_mask.reshape(np_with_mask.shape[0], 50*50)
    # Create numpy array with both arrays of data
    X = np.r_[np_with_mask, np_no_mask]
    # Create labels
    labels = np.zeros(X.shape[0])
    labels[200:] = 1.0
    names = {0: 'Mask', 1: 'No mask'}

    # Define train_size and test_size and train model
    x_train, x_test, y_train, y_test = train_test_split(X, labels, train_size=0.90, test_size=0.10)
    clf = make_pipeline(StandardScaler(), SVC(C=1.1, gamma='auto')) 
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    # Display accuracy score
    print("Accuracy Score: {:.2f}%".format(float(accuracy_score(y_test, y_pred)*100)))

    # Results array
    # Stores the predictions values in order to calculate confidence
    r = []

    while True:
        # Get each frame
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)

        # Convert frame to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert gray to binary black and white
        # Threshold values: 80 to 105
        _, bw_frame = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        # Detect face using Haar Cascade Classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
        faces_bw = face_cascade.detectMultiScale(bw_frame, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
        
        # Displays the number of detected faces
        cv2.putText(frame, str(len(faces)), (610,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Checks for detected faces
        if(len(faces) == 0 and len(faces_bw) == 0):
            cv2.putText(frame, "No face found...", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            # If the length of the array "face" is 0, then there is a face in black and white face array
            if len(faces) == 0:
                faces = faces_bw

            # Loops through all the faces
            for (x, y, w, h) in faces:
                # Trims the converted gray image to size of face
                face = gray[y:y+h, x:x+w]
                # Resizes de newly discovered face to 50x50
                face = cv2.resize(face, (50, 50))
                face = face.reshape(1,-1)
                # Makes a prediction with the newly discovered face
                pred = clf.predict(face)[0]
                # Gets the name corresponding with the prediction
                name = names[int(pred)]
                # Calculates and display the confidence value
                cv2.putText(frame, confidence(r)[1], (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # Evaluate if we can calculate confidence values
                # If there are not enough values, the result is based only in the prediction
                if len(r) <= 10:
                    if int(pred) == 0.0:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
                    else:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, name, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                # If there are values, calculate confidence values and display results
                else:
                    if confidence(r)[0] > 0.5:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, "Mask", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
                    else:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, "No mask", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

                r.append(pred)
                    
        # Show frame with results
        cv2.imshow('Realtime Facemask Detection', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    # Release video
    capture.release()
    cv2.destroyAllWindows()