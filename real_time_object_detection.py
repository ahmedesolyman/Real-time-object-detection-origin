# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import tensorflow as tf
import keras
from keras.utils import np_utils
from skimage.transform import resize

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
#CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
#	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
#	"sofa", "train", "tvmonitor"]
CLASSES = ['Corn_Downy mildew','Corn_Southern rust','Grape_Esca','Wheat_Brown rust',
 'Corn_Eyespot', 'Grape_Healthy', 'Wheat_Healthy', 'Corn_Healthy','Grape_Black rot','Grape_Powdery mildew','Wheat_Powdery mildew',
'Corn_Northern leaf blight','Grape_Chlorosis', 'Wheat_Black chaff','Wheat_Yellow rust']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
#net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
net=tf.keras.models.load_model('pdd_classifier_all_100x100_epochs_4.h5')
net.summary()
dummy_x = np.random.uniform(0,1,(1,256,256,3))
dummy_pred = net.predict( dummy_x )

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()

#	frame = imutils.resize(frame, width=400)
	frame_resized = resize( frame, output_shape = (1,256,256,3))
	# grab the frame dimensions and convert it to a blob
#	(h, w) = frame.shape[:2]
    
#	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
#		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
#	net.setInput(blob)
#	detections = net.forward()
	y_prediction = net.predict(frame_resized)
	# loop over the detections
	for index, prediction_prob in enumerate( y_prediction[0] ):
		# extract the confidence (i.e., probability) associated with
		# the prediction
        

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if prediction_prob > 0.1:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
	
			box = (0, 0, 256,256)
			# draw the prediction on the frame
			label = "{}: {:.2f}%".format( CLASSES[index],
				prediction_prob * 100)
			cv2.rectangle(frame, (0, 0), (256,256), COLORS[index], 2)
			y = 15
			cv2.putText(frame, label, (0, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[index], 2)

	# show the output frame
	cv2.imshow("Frame", frame_resized )
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
