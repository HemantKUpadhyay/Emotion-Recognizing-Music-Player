#Import required modules
import cv2
import dlib
import numpy as np
import pickle
from sklearn.svm import SVC
import math
import os
import PIL
from tkinter.filedialog import askdirectory
from PIL import ImageTk, Image
import pygame
from mutagen.id3 import ID3
from tkinter import *
import mutagen

video_capture = cv2.VideoCapture(0) #Webcam object
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)

    detections = detector(clahe_image, 1) #Detect the faces in the image

    for k,d in enumerate(detections): #For each detected face
        
        shape = predictor(clahe_image, d) #Get coordinates
        for i in range(1,68): #There are 68 landmark points on each face
                cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame

    cv2.imshow("image", frame) #Display the frame

    if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
        cv2.imwrite('pic.jpg',frame)
        break
video_capture.release()
cv2.destroyAllWindows()

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords

#emotions = [ "happiness", "neutral", "sadness"] #Emotion list
emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
# load the model from disk
'''loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)'''
#Set up some required objects
clf = pickle.load(open('finalized_model.sav','rb'))
frame = cv2.imread("pic.jpg") #Webcam object
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file


gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_image = clahe.apply(gray)

detections = detector(clahe_image, 1) #Detect the faces in the image
    #print(list(detections))

for k,d in enumerate(detections): #For each detected face
        
        shape = predictor(clahe_image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
            
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))

        arr = landmarks_vectorised
        arr = np.array(arr)
        arr =  arr.reshape(1, -1)
        pred=clf.predict(arr)
        print(pred[0])
        answer=""
        for i in pred:
            answer=emotions[i]
            print(answer)
        
        #Get coordinates
        #print(list(shape))
        '''for i in range(1,68): #There are 68 landmark points on each face
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame
        cv2.imshow("image", frame) #Display the frame'''
            
        if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
            
            break
        
print(answer)
if answer == "sadness":
    root = Tk()

    #This creates the main window of an application
    
    root.title("Emotion Recoginzing Music Player")
    root.geometry("600x300")
    root.configure(background='grey')
    Label(root, 
		 text="current playing song:",
		 fg = "red",
		 font = "Times").pack(side= LEFT, pady=100)


    listofsongs = []
    #realnames = []
    
    v = StringVar()
    songlabel = Label(root,textvariable=v,width=35)
    songlabel.pack(side=LEFT)
    
    index = 0
    directory ='E:/backup/New folder/test/sad'
    os.chdir(directory)
    for files in os.listdir(directory):
            if files.endswith(".mp3"):
    
                realdir = os.path.realpath(files)
                #audio = ID3(realdir)
                #realnames.append(audio['TIT2'].text[0])
    
    
                listofsongs.append(files)
    
    
            pygame.mixer.init()
            pygame.mixer.music.load(listofsongs[0])
            pygame.mixer.music.play()
    
    
    def updatelabel():
        global index
        global songname
        v.set(listofsongs[index])
        #return songname
    
    
    
    def nextsong(event):
        global index
        index += 1
        pygame.mixer.music.load(listofsongs[index])
        pygame.mixer.music.play()
        updatelabel()
    
    def prevsong(event):
        global index
        index -= 1
        pygame.mixer.music.load(listofsongs[index])
        pygame.mixer.music.play()
        updatelabel()
    
    
    def stopsong(event):
        pygame.mixer.music.stop()
        v.set("")
        #return songname
    
    
    label = Label(root,text='Mood Recognised : SADNESS',
		 fg = "red",
		 font = "Times")
    label.pack()
    
    listbox = Listbox(root)
    listbox.pack(side=TOP)
    
    #listofsongs.reverse()
    #realnames.reverse()
    
    for items in listofsongs:
        listbox.insert(0,items)
    
    #realnames.reverse()
    #listofsongs.reverse()
    
    
    nextbutton = Button(root,text = 'Next Song')
    nextbutton.pack(side=RIGHT)
    
    previousbutton = Button(root,text = 'Previous Song')
    previousbutton.pack(side=LEFT)
    
    stopbutton = Button(root,text='Stop Music')
    stopbutton.pack(side=BOTTOM)
    
    
    nextbutton.bind("<Button-1>",nextsong)
    previousbutton.bind("<Button-1>",prevsong)
    stopbutton.bind("<Button-1>",stopsong)
    
    
    root.mainloop()
    
if answer == "neutral":
    root = Tk()

    #This creates the main window of an application
    
    root.title("Emotion Recoginzing Music Player")
    root.geometry("600x300")
    root.configure(background='grey')
    Label(root, 
		 text="current playing song:",
		 fg = "red",
		 font = "Times").pack(side= LEFT, pady=100)


    listofsongs = []
    #realnames = []
    
    v = StringVar()
    songlabel = Label(root,textvariable=v,width=35)
    songlabel.pack(side=LEFT)
    
    index = 0
    directory ='E:/backup/New folder/test/neutral'
    os.chdir(directory)
    for files in os.listdir(directory):
            if files.endswith(".mp3"):
    
                realdir = os.path.realpath(files)
                #audio = ID3(realdir)
                #realnames.append(audio['TIT2'].text[0])
    
    
                listofsongs.append(files)
    
    
            pygame.mixer.init()
            pygame.mixer.music.load(listofsongs[0])
            pygame.mixer.music.play()
    
    
    def updatelabel():
        global index
        global songname
        v.set(listofsongs[index])
        #return songname
    
    
    
    def nextsong(event):
        global index
        index += 1
        pygame.mixer.music.load(listofsongs[index])
        pygame.mixer.music.play()
        updatelabel()
    
    def prevsong(event):
        global index
        index -= 1
        pygame.mixer.music.load(listofsongs[index])
        pygame.mixer.music.play()
        updatelabel()
    
    
    def stopsong(event):
        pygame.mixer.music.stop()
        v.set("")
        #return songname
    
    
    label = Label(root,text='Mood Recognised : NEUTRAL',
		 fg = "red",
		 font = "Times")
    label.pack()
    
    listbox = Listbox(root)
    listbox.pack(side=TOP)
    
    #listofsongs.reverse()
    #realnames.reverse()
    
    for items in listofsongs:
        listbox.insert(0,items)
    
    #realnames.reverse()
    #listofsongs.reverse()
    
    
    nextbutton = Button(root,text = 'Next Song')
    nextbutton.pack(side=RIGHT)
    
    previousbutton = Button(root,text = 'Previous Song')
    previousbutton.pack(side=LEFT)
    
    stopbutton = Button(root,text='Stop Music')
    stopbutton.pack(side=BOTTOM)
    
    
    nextbutton.bind("<Button-1>",nextsong)
    previousbutton.bind("<Button-1>",prevsong)
    stopbutton.bind("<Button-1>",stopsong)
    
    
    root.mainloop()
    
if answer == "happiness":
    root = Tk()

    #This creates the main window of an application
    
    root.title("Emotion Recoginzing Music Player")
    root.geometry("600x300")
    root.configure(background='grey')
    Label(root, 
		 text="current playing song:",
		 fg = "red",
		 font = "Times").pack(side= LEFT, pady=100)


    listofsongs = []
    #realnames = []
    
    v = StringVar()
    songlabel = Label(root,textvariable=v,width=35)
    songlabel.pack(side=LEFT)
    
    index = 0
    directory ='E:/backup/New folder/test/happy'
    os.chdir(directory)
    for files in os.listdir(directory):
            if files.endswith(".mp3"):
    
                realdir = os.path.realpath(files)
                #audio = ID3(realdir)
                #realnames.append(audio['TIT2'].text[0])
    
    
                listofsongs.append(files)
    
    
            pygame.mixer.init()
            pygame.mixer.music.load(listofsongs[0])
            pygame.mixer.music.play()
    
    
    def updatelabel():
        global index
        global songname
        v.set(listofsongs[index])
        #return songname
    
    
    
    def nextsong(event):
        global index
        index += 1
        pygame.mixer.music.load(listofsongs[index])
        pygame.mixer.music.play()
        updatelabel()
    
    def prevsong(event):
        global index
        index -= 1
        pygame.mixer.music.load(listofsongs[index])
        pygame.mixer.music.play()
        updatelabel()
    
    
    def stopsong(event):
        pygame.mixer.music.stop()
        v.set("")
        #return songname
    
    
    label = Label(root,text='Mood Recognised : HAPPINESS',
		 fg = "red",
		 font = "Times")
    label.pack()
    
    listbox = Listbox(root)
    listbox.pack(side=TOP)
    
    #listofsongs.reverse()
    #realnames.reverse()
    
    for items in listofsongs:
        listbox.insert(0,items)
    
    #realnames.reverse()
    #listofsongs.reverse()
    
    
    nextbutton = Button(root,text = 'Next Song')
    nextbutton.pack(side=RIGHT)
    
    previousbutton = Button(root,text = 'Previous Song')
    previousbutton.pack(side=LEFT)
    
    stopbutton = Button(root,text='Stop Music')
    stopbutton.pack(side=BOTTOM)
    
    
    nextbutton.bind("<Button-1>",nextsong)
    previousbutton.bind("<Button-1>",prevsong)
    stopbutton.bind("<Button-1>",stopsong)
    
    
    root.mainloop()
            
    
