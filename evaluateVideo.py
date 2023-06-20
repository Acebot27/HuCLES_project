'''
from keras.models import model_from_json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd


def load_model(path):

	json_file = open(path + 'model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	
	model = model_from_json(loaded_model_json)
	model.load_weights(path + "model.h5")
	print("Loaded model from disk")
	return model
	
def predict_emotion(gray, x, y, w, h):
	face = np.expand_dims(np.expand_dims(np.resize(gray[y:y+w, x:x+h]/255.0, (48, 48)),-1), 0)
	prediction = model.predict([face])

	return(int(np.argmax(prediction)), round(max(prediction[0])*100, 2))
	
path = "./model/"
model = load_model(path)

# face cascading file import
fcc_path = "./code_scripts/Tools/haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(fcc_path)
emotion_dict = {0: "Over Confident", 1: "Bit Nervous", 2: "Under Confident", 3: "Confident", 4: "Nervous", 5: "Bit Confident", 6: "Neutral"}
colour_cycle = ((0, 0, 255), (0, 85, 170), (255, 0, 0), (0, 255, 0), (170, 85, 0), (85, 170, 0), (0, 170, 85))

# results storing variables
videoC = []

# web cam or video processing 
webcam = cv2.VideoCapture(0)


while True:
	ret, frame = webcam.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	faces = faceCascade.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(30, 30)
									)
								
	for (count,(x, y, w, h)) in enumerate(faces):
		emotion_id, confidence = predict_emotion(gray, x, y, w, h)

		colour = colour_cycle[int(emotion_id)]
		# colour = colour_cycle[int(count%len(colour_cycle))]
		cv2.rectangle(frame, (x, y), (x+w, y+h), colour, 2)
		cv2.line(frame, (x+5, y+h+5),(x+100, y+h+5), colour, 20)
		cv2.putText(frame, "Face #"+str(count+1), (x+5, y+h+11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)

		cv2.line(frame, (x+8, y),(x+150, y), colour, 20)
		
		emotion = emotion_dict[emotion_id]

		# storing confi of faces frame wise for the video
		videoC.append(int(emotion_id))

		cv2.putText(frame, emotion + ": " + str(confidence) + "%" , (x+20, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)
	
	cv2.imshow('Confidence Evaluation - Press q to exit.', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'): break

webcam.release()
cv2.destroyAllWindows()

# plotting entire video frame
col = len(videoC)
img = np.zeros((50,col,3), np.uint8)

x = 0
for cID in videoC:
	colour = colour_cycle[int(cID)]
	cv2.line(img,(x,0),(x,49), colour, 3)
	x = x + 1

cv2.imwrite('./OutputImg/colorbar.png', img)
cv2.imshow('Color Map...', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('./testImg.jpg',cv2.IMREAD_COLOR)
# img = cv2.imread('./body/bdata/frames/BLITips/BLITips_frame_400.jpg',cv2.IMREAD_COLOR)
img0 = cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA)

face_cascade = cv2.CascadeClassifier('./code_scripts/Tools/haarcascade_frontalface_default.xml')
low_cascade = cv2.CascadeClassifier('./code_scripts/Tools/haarcascade_upperbody.xml')

gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY )

low = low_cascade.detectMultiScale(gray, 1.01 , 6, minSize=(30, 30))
faces = face_cascade.detectMultiScale(gray, 1.1 , 5)

    
for (x,y,w,h) in faces:
    cv2.rectangle(img0, (x,y), (x+w, y+h), (12,150,100),2)
for (x,y,w,h) in low:
    cv2.rectangle(img0, (x,y), (x+w, y+h), (12,150,100),2)
    
cv2.imshow('image',img0)
cv2.waitKey(0) # If you don'tput this line,thenthe image windowis just a flash. If you put any number other than 0, the same happens.
cv2.destroyAllWindows()