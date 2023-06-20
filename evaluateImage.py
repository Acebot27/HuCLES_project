from keras.models import model_from_json
import numpy as np
import cv2

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

fcc_path = "./code_scripts/Tools/haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(fcc_path)
emotion_dict = {0: "Over Confident", 1: "Bit Nervous", 2: "Under Confident", 3: "Confident", 4: "Nervous", 5: "Bit Confident", 6: "Neutral"}
colour_cycle = ((0, 0, 255), (0, 85, 170), (255, 0, 0), (0, 255, 0), (170, 85, 0), (85, 170, 0), (0, 170, 85))
# webcam = cv2.VideoCapture(0)  'angry.jpg' 'happy.jpg' 'sad.jpg' fear.jpg'
img = cv2.imread('images/angry1.jpg', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# while True:
# 	ret, frame = webcam.read()
# 	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
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
	cv2.rectangle(img, (x, y), (x+w, y+h), colour, 2)
	cv2.line(img, (x+5, y+h+5),(x+100, y+h+5), colour, 20)
	cv2.putText(img, "Face #"+str(count+1), (x+5, y+h+11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)

	cv2.line(img, (x+8, y),(x+150, y), colour, 20)
	
	emotion = emotion_dict[emotion_id]
	cv2.putText(img, emotion + ": " + str(confidence) + "%" , (x+27, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)

cv2.imshow('Confidence Evaluation - Press q to exit.', img)
# if cv2.waitKey(1) & 0xFF == ord('q'): break
cv2.waitKey(0)
# webcam.release()
cv2.destroyAllWindows()