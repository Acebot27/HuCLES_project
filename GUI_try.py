###############################################################################################################################
# Required libraries for Tkinter GUI
from tkinter import *
from tkinter import  ttk
from tkinter import filedialog
import cv2
import time
import math
import PIL.Image as Image
import PIL.ImageTk as ImageTk

# Required libraries for classification model and video processing
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt

###############################################################################################################################
Custom_Title_Font = ("Verdana", 20)
Custom_Section_Font = ("Times New Roman", 15)
Custom_Label_Font = ("Times New Roman", 12)
confi_dict = {0: "Under Confident", 1: "Bit Nervous", 2: "Over Confident", 3: "Confident", 4: "Nervous", 5: "Bit Confident", 6: "Neutral"} # use for FER 2013 dataset
colour_cycle = ((255,0,125), (255,255,0), (0, 0, 255), (0,128,255), (255,0,0), (0,255,255), (0,165,0), (105,105,105))

newC_dict = {0: "Over Confident", 1: "Confident", 2: "Neutral", 3: "Nervous", 4: "Under Confident"}
newClr_cycle = ((0, 0, 255), (0,128,255), (0,165,0), (255,0,0), (255,0,125), (105,105,105))
###############################################################################################################################

# Video Processing Class ######################################################################################################
class MyVideoCapture:
	def __init__(self, video_source=0):
		# Open the video source
		self.vid = cv2.VideoCapture(video_source)
		if not self.vid.isOpened():
			raise ValueError("Unable to open video source", video_source)

		# Get video source width and height
		self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
		self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
		# print("width =",self.width,'height =',self.height)

		# ---------------------------------------------------------------------------------------------------------------------
		# Initializing Parameters for detection through model...
		
		# results storing variables
		self.videoC = []		# for face
		self.allCL = []			# overall
		self.pBar = []			# progress bar list

		# Trained Face Images parameters
		self.fwd = 48
		self.fht = 48
		fmodel_path = './model/'
		# fmodel_path = './model/fmodel/'		# new model
		self.fmodel = self.load_model(fmodel_path)
		# ******** face cascading file import
		fcc_path = "./code_scripts/Tools/haarcascade_frontalface_alt.xml"
		self.faceCascade = cv2.CascadeClassifier(fcc_path)

		# Trained Upper Body Images parameters (width, height, etc.)
		self.ubwd = 240		#480 # 720
		self.ubht = 240		#480 # 600  # 720
		bmodel_path = './model/upperbody/' + 'newtry/'
		self.bmodel = self.load_model(bmodel_path)
		# ********* upper body cascading file import
		# bcc_path = "./code_scripts/Tools/haarcascade_upperbody.xml"
		# self.bodyCascade = cv2.CascadeClassifier(bcc_path)


		# ---------------------------------------------------------------------------------------------------------------------

	def get_frame(self):
		if self.vid.isOpened():
			ret, frame = self.vid.read()

			if ret:
				frame = self.processImg(frame)
				# Return a boolean success flag and the current frame converted to BGR
				return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
			else:
				return (ret, None)
		else:
			return (None, None)

	# Release the video source when the object is destroyed
	def __del__(self):
		if self.vid.isOpened():
			self.vid.release()
			cv2.destroyAllWindows()

		return self.videoC, self.allCL

	# -------------------------------------------------------------------------------------------------------------------------
	# loading the trained model 
	def load_model(self, model_path):
		json_file = open(model_path + 'model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		
		model = model_from_json(loaded_model_json)
		model.load_weights(model_path + "model.h5")
		print("Loaded model from disk")
		return model

	# predicting class from the model
	def predict_CL(self, model, gray, x, y, w, h, dwd, dht):
		InReg = np.expand_dims(np.expand_dims(np.resize(gray[y:y+w, x:x+h]/255.0, (dwd, dht)),-1), 0)
		prediction = model.predict([InReg])
		return(int(np.argmax(prediction)), round(max(prediction[0])*100, 2))

	# evaluating images frames...
	def processImg(self, imgFrame):
		gray = cv2.cvtColor(imgFrame, cv2.COLOR_BGR2GRAY)

		# ...............................................................................................................................
		# detect faces...
		faces = self.faceCascade.detectMultiScale(
				gray,
				scaleFactor=1.1,
				minNeighbors=5,
				minSize=(30, 30)
										)

		if faces == ():			# will change to <if not confi_id>
			self.pBar.append(5)
		'''
		else:
			# do UB detection and analysis
			ix = 175
			iy = 90
			iw = 1280 - 350			# imgWD - 2*175
			ih = 720 - 90			# imgHT - 90
			ubCID, ubScore = self.predict_CL(self.bmodel, gray, ix, iy, iw, ih, self.ubwd, self.ubht)
			# bclr = colour_cycle[int(ubCID)]
			# cv2.rectangle(imgFrame, (ix, iy), (ix+iw, iy+ih), bclr, 2)
			# cv2.line(imgFrame, (ix+8, iy),(ix+150, iy), bclr, 20)
			# confi_UB = confi_dict[ubCID]

			# storing confi of faces frame wise for the video
			self.allCL.append(int(ubCID))
			self.pBar.append(int(ubCID))
			# cv2.putText(imgFrame, confi_UB + ": " + str(ubScore) + "%" , (x+20, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)
		'''
		
		# for faces.....							
		for (count,(x, y, w, h)) in enumerate(faces):
			confi_id, C_score = self.predict_CL(self.fmodel, gray, x, y, w, h, self.fwd, self.fht)

			if confi_id == 5:
				confi_id = 3
			elif confi_id == 1:
				confi_id = 4
			colour = colour_cycle[int(confi_id)]
			# colour = colour_cycle[int(count%len(colour_cycle))]
			cv2.rectangle(imgFrame, (x, y), (x+w, y+h), colour, 2)
			cv2.line(imgFrame, (x+5, y+h+5),(x+100, y+h+5), colour, 20)
			cv2.putText(imgFrame, "Face #"+str(count+1), (x+5, y+h+11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)

			cv2.line(imgFrame, (x+8, y),(x+150, y), colour, 20)
			
			confi_L = confi_dict[confi_id]

			# storing confi of faces frame wise for the video
			self.videoC.append(int(confi_id))
			# self.pBar.append(int(confi_id))

			cv2.putText(imgFrame, confi_L + ": " + str(C_score) + "%" , (x+20, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)

			
			# do UB detection and analysis
			ix = x + int(w/2) - 360 
			iy = 1
			iw = 720
			ih = 717 - iy
			ubCID, ubScore = self.predict_CL(self.bmodel, gray, ix, iy, iw, ih, self.ubwd, self.ubht)
			# if ubCID == 5:
			# 	ubCID = 3
			# elif ubCID == 1:
			# 	ubCID = 4
			bclr = newClr_cycle[int(ubCID)]
			cv2.rectangle(imgFrame, (ix, iy), (ix+iw, iy+ih), bclr, 2)
			cv2.line(imgFrame, (ix+8, iy),(ix+200, iy+10), bclr, 20)
			confi_UB = newC_dict[ubCID]

			# storing confi of faces frame wise for the video
			self.allCL.append(int(ubCID))
			self.pBar.append(int(ubCID))
			cv2.putText(imgFrame, confi_UB + ": " + str(ubScore) + "%" , (ix+20, iy+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)
			


		# progress Bar creation ....................................................................................................................................
		
		# progress bar will be displayed 4% from the bottom of the frame
		y = math.ceil(imgFrame.shape[0] - imgFrame.shape[0]/25)
		# print ("\n################\nY:", y)

		# display progress bar across the width of the video
		wframe = imgFrame.shape[1]
		x = 0

		# white line as BG for progress bar...
		cv2.line(imgFrame, (x,y), (wframe,y), (255,255,255), 2)

		# displaying colorbar as progressbar....
		for cID in self.pBar:
			fclr = newClr_cycle[int(cID)]
			cv2.line(imgFrame, (x,y), (math.ceil(x),y+3), fclr, 5)
			x = x + 1
		# ..........................................................................................................................................................
		
		return imgFrame


	# -------------------------------------------------------------------------------------------------------------------------
###############################################################################################################################

# HuCLES APP Class ############################################################################################################
class HuCLESapp(Tk):
	def __init__(self, *args, **kwargs):

		Tk.__init__(self, *args, **kwargs)

		container = Frame(self)
		container.pack(side=TOP, fill=BOTH, expand=True)

		container.grid_rowconfigure(0, weight=1)
		container.grid_columnconfigure(0, weight=1)

		self.frames = {}

		for F in (StartPage, VideoPopUp, LinkingFrame):   # LinkingFrame
			frame = F(container, self)
			self.frames[F] = frame
			frame.grid(row=0, column=0, sticky=NSEW)

		self.show_frame(StartPage)

	
	def show_frame(self, container1, arg=None, arg0=None):
		frame = self.frames[container1]
		frame.tkraise()

		if arg:
			frame.loadSection(self, attr=arg, battr=arg0)
		elif arg == 0:
			frame.loadSection(self)

###############################################################################################################################

# Global Methods.... ##########################################################################################################
def app_exit():
	exit()

###############################################################################################################################

# Home (or Start) Page Class ##################################################################################################
class StartPage(Frame):
	def __init__(self, parent, controller):
		Frame.__init__(self, parent)

		FTitle  = Label(self, text="Welcome to HuCLES System!!!", font=Custom_Title_Font)
		FTitle.pack(padx=10, pady=20, fill=BOTH)

		sepTT = ttk.Separator(self, orient=HORIZONTAL)
		sepTT.pack(anchor=NW, fill=X)

		# -- Choice Selection Area & Buttons ----------------------------------------------------------------------------------
		self.lblTitle  = Label(self, text="Choose the way to feed video to the system for processing ...", font=Custom_Section_Font)
		self.lblTitle.pack(side=TOP, padx=10, pady=10, anchor=NW)

		# WebCam Button
		self.camBtn = Button(self, text="Use WebCam",
				command=lambda: controller.show_frame(VideoPopUp, 0))
		self.camBtn.pack(anchor=NW, ipadx=5, ipady=5, padx=30, pady=10)

		# Upload Video File Button 
		self.upldBtn = Button(self, text="Browse A FILE",
				command=lambda: self.browseFILE(controller))
		self.upldBtn.pack(anchor=NW, ipadx=5, ipady=5, padx=30, pady=10)


		# ---------------------------------------------------------------------------------------------------------------------

		# -- Exit or Quit and Refresh Application Button ----------------------------------------------------------------------------------
		quitBtn = Button(self, text="EXIT",
				command=lambda: app_exit(),
				width=7, height=3)
		quitBtn.place(anchor=NE, relx=1, x=-12, y=12)

		refreshBtn = Button(self, text='REFRESH',
					command=lambda: restartAPP(),
					width=11, height=3)
		refreshBtn.place(anchor=NW, x=12, y=12)
		# ---------------------------------------------------------------------------------------------------------------------

	# -- Browsing or selecting a video file from PC ... -----------------------------------------------------------------------
	def browseFILE(self, controller1):
		# Extracting a file name
		self.fname = filedialog.askopenfilename(initialdir="./", title='Select A Video File...',
					filetype=(("MP4", '*.mp4'), ("All Files", "*.*")))
		# print ("Inside browse button:",self.fname)
		self.dispFname = Label(self, text="")
		self.dispFname.place(x=140, y=200)
		self.dispFname.configure(text="Selected File is -> "+self.fname)
		controller1.show_frame(VideoPopUp, self.fname)

	# -------------------------------------------------------------------------------------------------------------------------

	# plotting entire Video detected frames result ----------------------------------------------------------------------------
	def plotVideoCL(self, clList, c_cycle, datatype=0):
		if clList:	
			cbPATH = './OutputImg/colorbar_{}.png'.format(datatype)
			lenObjs = len(clList)
			# print(lenObjs)

			img = np.zeros((30,lenObjs,3), np.uint8)

			x = 0
			for cID in clList:
				# print (int(cID))
				colour = c_cycle[int(cID)]
				cv2.line(img, (x,0), (x,29), colour, 20)
				x = x + 1

			cv2.imwrite(cbPATH, img)
			return cbPATH
	
	# -------------------------------------------------------------------------------------------------------------------------

	# plotting percentages of each confidence level ---------------------------------------------------------------------------
	def extractEachCL(self, conf, clList, c_cycle, datatype=0):
		if clList:
			opPATH = './OutputImg/CB{}_{}.png'.format(datatype, conf)		# datatype = (0 - face || 1 - wholeUB)....
			totalCLs = len(clList)
			cLevel = clList.count(conf)
			cPercent = cLevel / totalCLs * 100
			cColor = c_cycle[conf]

			img = np.zeros((15, totalCLs, 3), np.uint8)
			for x in range(totalCLs):
				if x <= int(cLevel):
					cv2.line(img, (x,0), (x,14), cColor, 20)
				else:
					cv2.line(img, (x,0), (x,14), (105,105,105), 20)

			cv2.imwrite(opPATH, img)
			return opPATH, cPercent, cLevel

	def plotPIEChart(self, valuesCL, otype):
		plt.clf()
		opfile = './OutputImg/pieChart_{}.png'.format(otype)
		CLlabels = ['Over Confident', 'Confident', 'Neutral', 'Nervous', 'Under Confident']
		CLcolors = ['red', 'darkorange', 'green', 'blue', 'purple']
		explode = (0,0.1,0,0,0)

		plt.pie(valuesCL, explode=explode, labels=CLlabels, 
				colors=CLcolors, shadow=True,
				startangle=90, autopct='%.2f%%')

		# plt.show()
		plt.savefig(opfile, transparent=True)
		return opfile

	# -------------------------------------------------------------------------------------------------------------------------

	def goToMCaN(self, controller1):
		faceperList = self.percList0
		nameList = ['OC', 'CO', 'NU', 'NV', 'UC']
		imax = 0
		for i in range(len(faceperList)-1):
			if faceperList[imax] < faceperList[i+1]:
				imax = i+1

		controller1.show_frame(LinkingFrame, self.rList, nameList[imax])
	
	# -- Result (Graphs and visualization) Section ----------------------------------------------------------------------------
	def loadSection(self, ctrlor, attr=None, battr=None):
		# self.camBtn.pack_forget()
		# self.upldBtn.pack_forget()


		sepTT = ttk.Separator(self, orient=HORIZONTAL)
		sepTT.pack(anchor=NW, fill=X)

		self.rList = attr			# face detection result list
		self.oList = battr
		# print (self.rList)

		# Button to go to M-CaN Window
		self.goBtn = Button(self, text="Open Syncergy",
				command=lambda: self.goToMCaN(ctrlor))
		self.goBtn.pack(ipadx=5, ipady=5, padx=5, pady=10)

		# Result Section 1 ********************************************************************************
		self.sec1 = LabelFrame(self, text="Result PART 1")
		self.sec1.pack(side=LEFT, fill=BOTH, expand=TRUE, padx=3, pady=2)

		# Contents in Section 1
		self.canvas1 = Canvas(self.sec1)
		self.canvas1.pack(pady=(15,2), fill=BOTH, expand=True)

		# Video Frames ColorBar 
		self.demo1 = Label(self.sec1, text="Body Parameters: ",
					font=Custom_Label_Font)
		self.demo1.place(anchor=NW, x=10, y=35)

		self.cbPath = self.plotVideoCL(self.oList, newClr_cycle, datatype=1)		# use oList here
		self.cbIMG = ImageTk.PhotoImage(file=self.cbPath)
		self.canvas1.create_image(150, 15, image=self.cbIMG, anchor=NW)
		
		# OC Percentage Bar...... 
		self.OCpath, self.OCcount, self.OCperc = self.extractEachCL(0, self.oList, newClr_cycle, datatype=1)	# oList
		self.labelOC = Label(self.sec1, text="Over Confident: ",
					font=Custom_Label_Font)
		self.labelOC.place(anchor=NW, x=10, y=85)
		self.OCIMG = ImageTk.PhotoImage(file=self.OCpath)
		self.canvas1.create_image(150, 75, image=self.OCIMG, anchor=NW)

		# CO Percentage Bar...... 
		self.COpath, self.COcount, self.COperc = self.extractEachCL(1, self.oList, newClr_cycle, datatype=1)	# oList
		self.labelCO = Label(self.sec1, text="Confident: ",
					font=Custom_Label_Font)
		self.labelCO.place(anchor=NW, x=10, y=115)
		self.COIMG = ImageTk.PhotoImage(file=self.COpath)
		self.canvas1.create_image(150, 105, image=self.COIMG, anchor=NW)
		
		# NU Percentage Bar...... 
		self.NUpath, self.NUcount, self.NUperc = self.extractEachCL(2, self.oList, newClr_cycle, datatype=1)	# oList
		self.labelNU = Label(self.sec1, text="Neutral: ",
					font=Custom_Label_Font)
		self.labelNU.place(anchor=NW, x=10, y=145)
		self.NUIMG = ImageTk.PhotoImage(file=self.NUpath)
		self.canvas1.create_image(150, 135, image=self.NUIMG, anchor=NW)
		
		# NV Percentage Bar...... 
		self.NVpath, self.NVcount, self.NVperc = self.extractEachCL(3, self.oList, newClr_cycle, datatype=1)	# oList
		self.labelNV = Label(self.sec1, text="Nervous: ",
					font=Custom_Label_Font)
		self.labelNV.place(anchor=NW, x=10, y=175)
		self.NVIMG = ImageTk.PhotoImage(file=self.NVpath)
		self.canvas1.create_image(150, 165, image=self.NVIMG, anchor=NW)
		
		# UC Percentage Bar...... 
		self.UCpath, self.UCcount, self.UCperc = self.extractEachCL(4, self.oList, newClr_cycle, datatype=1)	# oList
		self.labelUC = Label(self.sec1, text="Under Confident: ",
					font=Custom_Label_Font)
		self.labelUC.place(anchor=NW, x=10, y=205)
		self.UCIMG = ImageTk.PhotoImage(file=self.UCpath)
		self.canvas1.create_image(150, 195, image=self.UCIMG, anchor=NW)

		
		# displaying PIE chart ....
		percList = [self.OCperc, self.COperc, self.NUperc, self.NVperc, self.UCperc]
		self.piePath = self.plotPIEChart(percList, 'UB')	# wholeUB replace with face
		self.pieImg = ImageTk.PhotoImage(file=self.piePath)
		self.canvas1.create_image(100, 215, image=self.pieImg, anchor=NW)
		
		# **************************************************************************************************

		# Section separator
		sep = ttk.Separator(self, orient=VERTICAL)
		sep.pack(side=LEFT, fill=Y, padx=(2,1))

		# Result Section 2 *********************************************************************************
		self.sec2 = LabelFrame(self, text="Result PART 2")
		self.sec2.pack(side=LEFT, fill=BOTH, expand=TRUE, padx=3, pady=2)

		# Contents in Section 2
		self.canvas2 = Canvas(self.sec2)
		self.canvas2.pack(pady=15, fill=BOTH, expand=True)

		self.demo2 = Label(self.sec2, text="Based on Face: ",
					font=Custom_Label_Font)
		self.demo2.place(anchor=NW, x=10, y=35)

		# Face part Output visualization
		self.cbPath0 = self.plotVideoCL(self.rList, colour_cycle)	
		self.cbIMG0 = ImageTk.PhotoImage(file=self.cbPath0)
		self.canvas2.create_image(150, 15, image=self.cbIMG0, anchor=NW)
		

		# OC Percentage Bar...... 
		self.OCpath0, self.OCcount0, self.OCperc0 = self.extractEachCL(2, self.rList, colour_cycle)
		self.labelOC0 = Label(self.sec2, text="Over Confident: ",
					font=Custom_Label_Font)
		self.labelOC0.place(anchor=NW, x=10, y=85)
		self.OCIMG0 = ImageTk.PhotoImage(file=self.OCpath0)
		self.canvas2.create_image(150, 75, image=self.OCIMG0, anchor=NW)

		# CO Percentage Bar...... 
		self.COpath0, self.COcount0, self.COperc0 = self.extractEachCL(3, self.rList, colour_cycle)
		self.labelCO0 = Label(self.sec2, text="Confident: ",
					font=Custom_Label_Font)
		self.labelCO0.place(anchor=NW, x=10, y=115)
		self.COIMG0 = ImageTk.PhotoImage(file=self.COpath0)
		self.canvas2.create_image(150, 105, image=self.COIMG0, anchor=NW)
		
		# NU Percentage Bar...... 
		self.NUpath0, self.NUcount0, self.NUperc0 = self.extractEachCL(6, self.rList, colour_cycle)
		self.labelNU0 = Label(self.sec2, text="Neutral: ",
					font=Custom_Label_Font)
		self.labelNU0.place(anchor=NW, x=10, y=145)
		self.NUIMG0 = ImageTk.PhotoImage(file=self.NUpath0)
		self.canvas2.create_image(150, 135, image=self.NUIMG0, anchor=NW)
		
		# NV Percentage Bar...... 
		self.NVpath0, self.NVcount0, self.NVperc0 = self.extractEachCL(4, self.rList, colour_cycle)
		self.labelNV0 = Label(self.sec2, text="Nervous: ",
					font=Custom_Label_Font)
		self.labelNV0.place(anchor=NW, x=10, y=175)
		self.NVIMG0 = ImageTk.PhotoImage(file=self.NVpath0)
		self.canvas2.create_image(150, 165, image=self.NVIMG0, anchor=NW)
		
		# UC Percentage Bar...... 
		self.UCpath0, self.UCcount0, self.UCperc0 = self.extractEachCL(0, self.rList, colour_cycle)
		self.labelUC0 = Label(self.sec2, text="Under Confident: ",
					font=Custom_Label_Font)
		self.labelUC0.place(anchor=NW, x=10, y=205)
		self.UCIMG0 = ImageTk.PhotoImage(file=self.UCpath0)
		self.canvas2.create_image(150, 195, image=self.UCIMG0, anchor=NW)

		
		# displaying PIE chart ....
		self.percList0 = [self.OCperc0, self.COperc0, self.NUperc0, self.NVperc0, self.UCperc0]
		self.piePath0 = self.plotPIEChart(self.percList0, 'face')
		self.pieImg0 = ImageTk.PhotoImage(file=self.piePath0)
		self.canvas2.create_image(100, 215, image=self.pieImg0, anchor=NW)
		
		# **************************************************************************************************
	# -------------------------------------------------------------------------------------------------------------------------

###############################################################################################################################

# Video Displaying Page Class #################################################################################################
class VideoPopUp(Frame):
	def __init__(self, parent, controller):
		Frame.__init__(self, parent)

		lblTitle  = Label(self, text="Analyzing the Video ...", font=Custom_Title_Font)
		lblTitle.pack(padx=10, pady=20, fill=BOTH)

		# self.video_source = video_source
		# self.loadVideo()

	# --------------------------------------------------------------------------------------------------------------------------

	# -- Video Canvas Visualization --------------------------------------------------------------------------------------------
	def loadSection(self, controller, attr=0, battr=None):
		# ****************************************************************************************
		sepTT = ttk.Separator(self, orient=HORIZONTAL)
		sepTT.pack(anchor=NW, fill=X)

		# ** Video Widget Initialization *********************************************************
		self.video_source = attr
		print (self.video_source)

		# Opening video source
		self.vid = MyVideoCapture(self.video_source)

		# Create a canvas that can fit the above video source size
		# self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
		self.canvas = Canvas(self, width = 960, height = 540)
		self.canvas.pack(pady=15)

		# Button that lets the user take a snapshot
		self.btn_ss = Button(self, text="Snapshot", width=50, height=2, command=self.snapshot)
		self.btn_ss.pack(anchor=CENTER, expand=True)

		# After it is called once, the update method will be automatically called every delay milliseconds
		self.delay = 2
		self.updateVidFrame()

		# Back Button
		bkBtn = Button(self, text="Stop & Go Back",
				command=lambda: self.stopVid(controller),
				width=13, height=3)
		bkBtn.place(anchor=NE, relx=1, x=-12, y=12)
				
	# ****************************************************************************************
	def snapshot(self):
		# Get a frame from the video source
		ret, frame = self.vid.get_frame()

		if ret:
			cv2.imwrite("./images/frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

	def updateVidFrame(self):
		# Get a frame from the video source
		ret, frame = self.vid.get_frame()

		if ret:
			self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame).resize((960,540)))
			self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

		self.after(self.delay, self.updateVidFrame)

	# ****************************************************************************************
	# Stopping the video and going back to main screen ...
	def stopVid(self, ctrlor):
		videoC, allCL = self.vid.__del__()
		self.arg = videoC # 'testing'
		self.arg0 = allCL
		ctrlor.show_frame(StartPage, self.arg, self.arg0)

	# --------------------------------------------------------------------------------------------------------------------------


###############################################################################################################################


# Mapping with Hormones Page Class #################################################################################################
class LinkingFrame(Frame):
	def __init__(self, parent, controller):
		Frame.__init__(self, parent)

		lblTitle  = Label(self, text="SYNCERGY (M - CaN)...", font=Custom_Title_Font)
		lblTitle.pack(padx=10, pady=20, fill=BOTH)

		# Back Button
		bkBtn = Button(self, text="Go Back",
				command=lambda: self.goBack(controller),
				width=10, height=3)
		bkBtn.place(anchor=NE, relx=1, x=-12, y=12)

		sepTT = ttk.Separator(self, orient=HORIZONTAL)
		sepTT.pack(anchor=NW, fill=X)

		self.canvasMN = Canvas(self, width = 960, height = 540)
		self.canvasMN.pack(pady=5, expand=True)

		self.imgMCaN = ImageTk.PhotoImage(file='./Syncergy.png')
		self.canvasMN.create_image(5, 5, image=self.imgMCaN, anchor=NW)



	# --------------------------------------------------------------------------------------------------------------------------

	# -- Syncergy Evaluation Visualization --------------------------------------------------------------------------------------------
	def loadSection(self, controller, attr=0, battr=None):
		# ****************************************************************************************
		sepTT = ttk.Separator(self, orient=HORIZONTAL)
		sepTT.pack(anchor=NW, fill=X)

		# ** MN Initialization *********************************************************
		self.fList = attr
		self.NE = 0.0
		self.DA = 0.0
		self.HT5 = 0.0

		# evaluate Button
		self.evalBtn = Button(self, text="Evaluate Syncergy",
				command=lambda: self.evalSyncergy(attr, battr))
		self.evalBtn.pack(ipadx=10, ipady=10, padx=5, pady=10)

				
	# ****************************************************************************************
	def evalSyncergy(self, vfList, lARG):
		# vfList - Contains confidence value results obtained for each frames 
		totalF = len(vfList) # total number of frames...
		
		# calculating amount of each MN based on the contents
		for xitem in vfList:
			self.NE += 0.1
			self.DA += 0.1
			self.HT5 += 0.1

			if xitem == 2:			# For OverConfident
				self.HT5 += 0.9
			elif xitem == 3:		# For Confident
				self.DA += 0.8
				self.HT5 += 0.3
			elif xitem == 6:		# For Neutral
				self.NE += 0.4
				self.DA += 0.4
				self.HT5 += 0.4
			elif xitem == 4:		# For Nervous
				self.NE += 0.55
				self.HT5 += 0.55
			elif xitem == 0:		# For UnderConfident
				self.NE += 0.9

		# Neutralizing each count and acquiring the percentage of each MN by dividing the counts by the total number of frames...
		tNE = self.NE / totalF
		tDA = self.DA / totalF
		tHT5 = self.HT5 / totalF

		sepTT = ttk.Separator(self, orient=HORIZONTAL)
		sepTT.pack(anchor=NW, fill=X)

		self.resultName = self.getResultStr(lARG)
		
		self.FinalLabel = Label(self, text="PREDICTED -  "+self.resultName, font=Custom_Section_Font)
		self.FinalLabel.pack(side=TOP, padx=5, pady=(10,5))

		# divide into two sections
		sepTT = ttk.Separator(self, orient=HORIZONTAL)
		sepTT.pack(anchor=NW, fill=X)

		# section 1
		self.Rsec1 = LabelFrame(self)
		self.Rsec1.pack(side=LEFT, fill=BOTH, expand=TRUE, padx=3, pady=2)

		self.canResults = Canvas(self.Rsec1)
		self.canResults.pack(pady=10, fill=BOTH, expand=True)

		self.pathNE = self.plotBarMN(0, tNE)
		self.labelNE = Label(self.canResults, text="Norepinephrine (NE): ",
					font=Custom_Label_Font)
		self.labelNE.place(anchor=NW, x=40, y=40)
		self.imgNE = ImageTk.PhotoImage(file=self.pathNE)
		self.canResults.create_image(220, 47, image=self.imgNE, anchor=NW)

		self.pathDA = self.plotBarMN(3, tDA)
		self.labelDA = Label(self.canResults, text="Dopamine (DA): ",
					font=Custom_Label_Font)
		self.labelDA.place(anchor=NW, x=40, y=70)
		self.imgDA = ImageTk.PhotoImage(file=self.pathDA)
		self.canResults.create_image(220, 77, image=self.imgDA, anchor=NW)

		self.pathHT5 = self.plotBarMN(2, tHT5)
		self.labelHT5 = Label(self.canResults, text="Serotonin (5-HT): ",
					font=Custom_Label_Font)
		self.labelHT5.place(anchor=NW, x=40, y=100)
		self.imgHT5 = ImageTk.PhotoImage(file=self.pathHT5)
		self.canResults.create_image(220, 107, image=self.imgHT5, anchor=NW)

		# section 2
		# Section separator
		sep = ttk.Separator(self, orient=VERTICAL)
		sep.pack(side=LEFT, fill=Y, padx=(2,1))

		# Result Section 2 *********************************************************************************
		self.Rsec2 = LabelFrame(self)
		self.Rsec2.pack(side=LEFT, fill=BOTH, expand=TRUE, padx=3, pady=2)

		# Contents in Section 2
		self.canRes2 = Canvas(self.Rsec2)
		self.canRes2.pack(pady=2, fill=BOTH, expand=True)

		valMN = [tNE, tDA, tHT5]
		self.MNpiePath = self.plotDoughnut(valMN)
		self.imgMNPie = ImageTk.PhotoImage(file=self.MNpiePath)
		self.canRes2.create_image(300, 0, image=self.imgMNPie, anchor=NW)
		return


	def getResultStr(self, CLval):
		if CLval == 'OC':
			tempName = '## Over Confident ## - Since content of Norepinephrine AND Dopamine <<< Serotonin...'
		elif CLval == 'CO':
			tempName = '## Confident ## - Since content of Norepinephrine << Dopamine OR BOTH Dopamine AND Serotonin...'
		elif CLval == 'NU':
			tempName = '## Neutral ## - Since content of Norepinephrine, Dopamine AND Serotonin are around nearby to each other...'
		elif CLval == 'NV':
			tempName = '## Nervous ## - Since content of Norepinephrine AND Serotonin >> Dopamine...'
		elif CLval == 'UC':
			tempName = '## Under Confident ## - Since content of Norepinephrine >>> Serotonin AND Dopamine...'

		namelbl = tempName
		return namelbl

	def plotBarMN(self, cCode, mnPerc):
		opPATH = './OutputImg/MN_{}.png'.format(cCode)		# datatype = (0 - face || 1 - wholeUB)....
		totalFr = len(self.fList)
		
		bEnd = int(720 * mnPerc)
		print ("Neuro -",cCode,":",bEnd)
		cColor = newClr_cycle[cCode]

		img = np.zeros((15, 720, 3), np.uint8)
		for x in range(720):
			if x <= int(bEnd):
				cv2.line(img, (x,0), (x,14), cColor, 7)
			else:
				cv2.line(img, (x,0), (x,14), (105,105,105), 7)

		cv2.imwrite(opPATH, img)
		return opPATH

	
	# Plot Doughnut for Neurotransmitters...
	def plotDoughnut(self, valuesMN):
		plt.clf()
		nopfile = './OutputImg/NeuroPC.png'
		MNlabels = ['NE', 'DA', 'SER']
		MNcolors = ['red', 'blue', 'green']
		# explode = (0,0.1,0,0,0)
		# Create a circle for the center of the plot
		my_circle=plt.Circle( (0,0), 0.40, color='white')

		plt.pie(valuesMN, labels=MNlabels, 
				colors=MNcolors, shadow=True,
				startangle=90, autopct='%.2f%%')

		p=plt.gcf()
		p.gca().add_artist(my_circle)
		# plt.show()
		# plt.show()
		plt.savefig(nopfile, transparent=True, dpi=60)
		return nopfile


	# ****************************************************************************************
	# Stopping the video and going back to main screen ...
	def goBack(self, ctrlor):
		# videoC, allCL = self.vid.__del__()
		# self.arg = videoC # 'testing'
		# self.arg0 = allCL
		ctrlor.show_frame(StartPage)

	# --------------------------------------------------------------------------------------------------------------------------


###############################################################################################################################


# Creating App instance... ####################################################################################################
def initalizeAPP():
	global app
	app = HuCLESapp()
	app.geometry("1280x720+30+30")   # "960x540+50+50"  # -> " w x h + xOffset + yOffset "
	app.title("HuCLES-Home_Page")
	app.mainloop()

# Restarting or Refreshing APP
def restartAPP():
	app.destroy()
	initalizeAPP()

###############################################################################################################################

# Driver Code
if __name__ == '__main__':
	initalizeAPP()

###############################################################################################################################
