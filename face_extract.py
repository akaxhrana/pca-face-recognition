import os
import cv2
import numpy as np

# Define paths
base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + 'model_data/deploy.prototxt')
caffemodel_path = os.path.join(base_dir + 'model_data/weights.caffemodel')

# Read the model
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Loop through the data and get names
for person_name in os.listdir(base_dir + 'data/'):

	# Create directories
	if not os.path.exists('face_data'):
			os.makedirs('face_data')
		
	if not os.path.exists('face_data/' + person_name):
		os.makedirs('face_data/' + person_name)

	# Loop through the images
	for file in os.listdir(base_dir + 'data/' + person_name):

		file_name, file_extension = os.path.splitext(file)

		if (file_extension in ['.png','.jpg', '.jpeg']):
			print("Image path: {}".format(base_dir + 'data/' + person_name + '/' + file))


			image = cv2.imread(base_dir + 'data/'+ person_name + '/' + file)

			blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

			model.setInput(blob)
			detections = model.forward()

			(h, w) = image.shape[:2]

			# Getting faces
			for i in range(0, detections.shape[2]):
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				confidence = detections[0, 0, i, 2]
						
				# If confidence > 0.5, extract the face
				if (confidence > 0.5):
					frame = image[startY:endY, startX:endX]
					(h_, w_) = frame.shape[:2]

					# (62, 47) frame needed
					if(h_ > 62 and w_ >= 47):
						cv2.imwrite(base_dir + 'face_data/'+ person_name + '/' + file_name + '_' + str(i) + file_extension, frame)

		