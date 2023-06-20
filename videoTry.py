# Color Codes ##################################
'''
Red - 		(0,0,255)	- OC - 0    - red
Orange - 	(0,128,255)	- CO - 3, 5 - green
Yellow - 	(0,255,255)	- BC - 5-x  - orange
Green - 	(0,255,0)	- NU - 6    - orange
SkyBlue - 	(255,255,0)	- BN - 1-x  - blue
Blue - 		(255,0,0)	- NV - 4, 1 - blue
Purple - 	(255,0,125)	- UC - 2    - purple
'''
################################################
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
colour_cycle = ((0, 0, 255), (255,255,0), (255,0,125), (0,128,255), (255,0,0), (0,255,255), (0,255,0))

# results storing variables
videoC = []

# web cam or video processing 
webcam = cv2.VideoCapture(0)

fc = 0
while True:
	ret, frame = webcam.read()
	if fc % 1 == 0:
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
	fc += 1

webcam.release()
cv2.destroyAllWindows()

# plotting entire video frame ##########################################################
col = len(videoC)
img = np.zeros((30,col,3), np.uint8)

x = 0
for cID in videoC:
	colour = colour_cycle[int(cID)]
	cv2.line(img,(x,0),(x,29), colour, 8)
	x = x + 1

cv2.imwrite('./OutputImg/colorbar.png', img)
cv2.imshow('Color Map...', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
########################################################################################

# plotting percentages of each confidence level ########################################
totLevel = col

def plotCL(conf, videoC, colour_cycle):
	totLevel = len(videoC)
	cLevel = videoC.count(conf)
	cPercent = cLevel / totLevel * 100
	cColor = colour_cycle[conf]

	img0 = np.zeros((50,totLevel,3), np.uint8)
	for x in range(100):
		if x <= int(cPercent):
			cv2.line(img0,(x,0),(x,49),cColor, 9)
		else:
			cv2.line(img0,(x,0),(x,49),(105,105,105), 9)

	cv2.imshow('Demo Map...', img0)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return cPercent, cLevel


print("\n#############################")
print ("Total Frames:", totLevel)
OCperc, OCLevel = plotCL(0, videoC, colour_cycle)
print("#############################")
print ("OC-Frames:", OCLevel)
print ("OC % :",OCperc)
print("#############################")

BNperc, BNLevel = plotCL(1, videoC, colour_cycle)
print("#############################")
print ("BN-Frames:", BNLevel)
print ("BN % :",BNperc)
print("#############################")

UCperc, UCLevel = plotCL(2, videoC, colour_cycle)
print("#############################")
print ("UC-Frames:", UCLevel)
print ("UC % :",UCperc)
print("#############################")

COperc, COLevel = plotCL(3, videoC, colour_cycle)
print("#############################")
print ("CO-Frames:", COLevel)
print ("CO % :",COperc)
print("#############################")

NVperc, NVLevel = plotCL(4, videoC, colour_cycle)
print("#############################")
print ("NV-Frames:", NVLevel)
print ("NV % :",NVperc)
print("#############################")

BCperc, BCLevel = plotCL(5, videoC, colour_cycle)
print("#############################")
print ("BC-Frames:", BCLevel)
print ("BC % :",BCperc)
print("#############################")

NUperc, NULevel = plotCL(6, videoC, colour_cycle)
print("#############################")
print ("NU-Frames:", NULevel)
print ("NU % :",NUperc)
print("#############################")

########################################################################################
