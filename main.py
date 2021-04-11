import os
import cv2
import numpy as np
import glob

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# Define paths
base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + 'model_data/deploy.prototxt')
caffemodel_path = os.path.join(base_dir + 'model_data/weights.caffemodel')

# Read the model
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Loop through the data and get names
print("Extracting faces from original images")
for person_name in os.listdir(base_dir + 'data/'):

    # Create directories
    if not os.path.exists('face_data'):
        os.makedirs('face_data')

    if not os.path.exists('face_data/' + person_name):
        os.makedirs('face_data/' + person_name)

    # Loop through the images
    for file in os.listdir(base_dir + 'data/' + person_name):

        file_name, file_extension = os.path.splitext(file)

        if (file_extension in ['.png', '.jpg', '.jpeg']):
            print("Image path: {}".format(
                base_dir + 'data/' + person_name + '/' + file))

            image = cv2.imread(base_dir + 'data/' + person_name + '/' + file)

            # Generating blob from the images to feed to the CV2 deep neural network(DNN)
            blob = cv2.dnn.blobFromImage(cv2.resize(
                image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

            model.setInput(blob)
            detections = model.forward()

            (h, w) = image.shape[:2]

            # Getting faces
            for i in range(0, detections.shape[2]):
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                confidence = detections[0, 0, i, 2]

                # If confidence > 0.9, extract the face
                if (confidence > 0.9):
                    frame = image[startY:endY, startX:endX]
                    (h_, w_) = frame.shape[:2]

                    # (62, 47) frame needed
                    if(h_ > 62 and w_ >= 47):

                        # convert to gray scale
                        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        image_res = cv2.resize(image_gray, (47, 62))
                        cv2.imwrite(base_dir + 'face_data/' + person_name + '/' + file_name + '_' + str(i) + file_extension, image_res)


print("Extraction complete")


'''
        Creating a dataset from images
'''

dataset = []
dataset_labels = []
for person in os.listdir(base_dir + 'face_data/'):

    for file in os.listdir(base_dir + 'face_data/' + person):

        path = base_dir + 'face_data/' + person + '/'
        image = cv2.imread(path + file, 0)
        dataset.append(image)
        dataset_labels.append(person)

# Creating numpy array of images
dataset = np.array(dataset, dtype='float64')
dataset_labels = np.array(dataset_labels, dtype='str')
# Reshaping the array to (no. of rows, no. of columns)
dataset = np.reshape(dataset, [dataset.shape[0], dataset.shape[1]*dataset.shape[2]])

X = dataset
y = dataset_labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# Using pca to retian 90% variance of the data
pca = PCA(0.90).fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


print("Fitting the classifier to the training set")
clf = MLPClassifier(hidden_layer_sizes=(1024,), verbose=True,
                    early_stopping=True).fit(X_train_pca, y_train)


y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
