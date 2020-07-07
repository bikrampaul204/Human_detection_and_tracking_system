import cv2
import glob
import os
import time
import MySQLdb

import imutils
import argparse
import sys
import _thread
from imutils.object_detection import non_max_suppression
from mysql.connector import MySQLConnection, Error

x=15.0
y=20.0
db = MySQLdb.connect("localhost","root","Ki!!erStr1ke","logs" )
cursor = db.cursor()
counter=[0,0,0,0,0,0]#to keep a count and check how many trackers are there on the same person
person_counter=0#to keep a count on the no of people
names=["","","3","4","5"]#the names of the people that could be tracked
face_cascade1=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#face cascade for detecting face
upperBody_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')#upperbody cascade fro detectng upperbody
hog = cv2.HOGDescriptor()#for person detection
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())#people detector

recognizer = cv2.face.LBPHFaceRecognizer_create()#recognizing a face

camera =cv2.VideoCapture(0)#initializing thee camera

def assure_path_exists(path):#unit test this function to check if the folder exists
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        make_directory(dir)
def make_directory(dir):#unit test this function to check if a new directory is being crreated if the directory is not present
    os.makedirs(dir)
def clickImage():#unit test this function to check if image is being clicked
    ok, frame = camera.read()
    return ok,frame
def frameResize(frame):
    frame_resized = imutils.resize(frame, width=min(800, frame.shape[1]))#resize the frame
    return frame_resized
def grayScaleConverter(frame):#unit test this to check if gray scale is happening
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#gray scale the resized frame
    return gray
def personDetect(frame):#unit test this to check if person is being detected
  (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.06)#detect person
  rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)#form a rectangle around the detected person
  return rects

def upperBodyDetect(frame):#unit test this to check if upperbody is being detected
    arrUpperBody = upperBody_cascade.detectMultiScale(frame,scaleFactor = 1.2,minNeighbors = 5,minSize = (30,30),flags = cv2.CASCADE_SCALE_IMAGE)
    return arrUpperBody
  

def faceDetect(frame):#unit test this to check if face is being detected
    Face = face_cascade1.detectMultiScale(frame,scaleFactor = 1.2,minNeighbors = 5,minSize = (30,30),flags = cv2.CASCADE_SCALE_IMAGE)#face cascade
    return Face

def newThreadCreator(array):#unit test this to check if a thread is being created
    for (x, y, w, h) in array:
          bbox=(x,y,w,h)
          try:
           _thread.start_new_thread(tracking,(frame,bbox))#create thread if person detected
          except:
            print ("Error: unable to start thread")
def initiateTracker():#unit test it to check if the tracker is being initiated
    tracker= cv2.TrackerKCF_create() #initialize the tracker               
    return tracker
def getCurrentTime():#unit test this to check if the current time is being given
    return time.strftime("%H:%M:%S")
def getCurrentDate():#unit test this to check if the current time is being given
    return time.strftime("%d/%m/%Y")
def recognizePerson(gray,x,y,w,h):
    Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])#recognize the person, returns the Id and confidence             
    print(str(Id)+" "+str(confidence))#display the confidence
    return Id,confidence               
def tracking(frame,bbox):
    tracker=initiateTracker()
    tme=getCurrentTime()
    global person_counter#the global variable to keep track of the no of person in the room
    ok=tracker.init(frame,bbox)#allocate a tracker (modify)
    date=getCurrentDate()#store the current date
    gray=grayScaleConverter(frame)#convert into gray color
    check=0#see if once also the tracking has taken place
    flag=0#to store the id of the person
    name="Human"#default the person being tracked is "human"
    should_track=0#variable to store and the value determines if the tracking must happen or not
    while True:
        if (flag==0):#during the begin of tracking person unrecognized so flag=0
            ok,frame=clickImage()#read frames from the camera
            gray=grayScaleConverter(frame)#convert to gray color
            faceDetector = faceDetect(frame)#detect the no of faces
            try:
                ok,bbox=tracker.update(frame)#update the tracker(modify)
                if ok:
                    check=1;#as once tracking is possible so check=1
                    for (x, y, w, h) in faceDetector:#go through the faces that were detected
                        if x>=(bbox[0]-30) and y<=(bbox[1]+30) and w<=(bbox[2]+30) and h<=(bbox[3]+30):#check if the area being tracked and the area where the face is seen are same
                            Id, confidence = recognizePerson(gray,x,y,w,h)#recognize the person, returns the Id and confidenc
                            if round(confidence,2)>=55:#if confidence is greater than 60, then check among the Id's
                                flag=Id                                
                                name=names[flag-1]#associate the name to the person being tracked
                                if counter[flag-1]==0:#increase counter
                                    should_track=1
                                    counter[flag-1]=1
                                elif(counter[flag-1]==1):#if the person is already being tracked then 
                                    person_counter-=1
                                    break
            
                else :
                    end_time=getCurrentTime()
                    person_counter-=1
                    if check==1:#if atleast once deteted then display result
                        print("person"+str(name)+" was here from "+date+" time: "+str(tme)+" to "+str(end_time))
                    break
            except:
                print("Exception")
                person_counter-=1
                break
                        
            
        if(should_track==1):
            ok,frame=clickImage()
            try:
                ok,bbox=tracker.update(frame)
                if ok:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                    check=1;
                    
                else :
                    cv2.destroyWindow(name)#destroy the frame if fail to track
                    end_time=start_time=getCurrentTime()
                    counter[flag-1]=0
                    should_track=0
                    person_counter-=1
                    if check==1:#if atleast once deteted then display result
                        print("person"+str(name)+" was here from "+date+" time: "+str(tme)+" to "+str(end_time))
                        print ("ola -1")    
                        x_loc=234.5
                        y_loc=456.6
                        insert1(x_loc,y_loc,name,tme,end_time,date)    
                        print("ola+1")                
                    break
            except:
                print("exception 2")
                person_counter-=1
                break
            cv2.imshow(name, frame)#display the tracking box
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break
    print("i'm out")    
        
def insert1(xx,yy,ID,st,et,dt):
  sql="INSERT INTO activitylog(x,y,id,startime,endtime,date) VALUES(%s,%s,%s,%s,%s,%s)"
  args=(xx,yy,str(ID),str(st),str(et),str(dt))
  try:
    cursor.execute(sql,args)
    print("Ola")
    db.commit()
  except (MySQLdb.Error, MySQLdb.Warning) as e:
    print(e)
    db.rollback()
        
#function to detect face/upperbody and person and start tracking if possible
def detect_all(frame):  
  rects = personDetect(frame)
  personcount=0#a counter to count the no of people in the frame
  upperbodycount=0#a counter to count the no of upperbodies in the frame
  facecount=0#a counter to count the no of faces in the frame
  global person_counter
  for (x, y, w, h) in rects:#count the no of person in the frame
      personcount+=1
  if personcount>person_counter:#if there is an increase in the no of people then start tracking that person
      person_counter+=personcount-person_counter
      newThreadCreator(rects)
  arrUpperBody=upperBodyDetect(frame)
  for (x, y, w, h) in arrUpperBody:#count he no of upperbodies
      upperbodycount+=1
  if upperbodycount>person_counter:#if there is an increase in the no of upperbodies then start tracking that person
      person_counter+=upperbodycount-person_counter
      newThreadCreator(arrUpperBody)
  Face=faceDetect(frame)
  for (x, y, w, h) in Face:#count he no of faces
      facecount+=1
  if facecount>person_counter:#if there is an increase in the no of faces then start tracking that person
      person_counter+=facecount-person_counter
      newThreadCreator(Face)
  return frame

#this function is used for background subtraction, if there is no sufficient difference between the two images then they are considered as the same image
def background_subtraction(previous_frame, frame_resized_grayscale, min_area):
  frameDelta = cv2.absdiff(previous_frame, frame_resized_grayscale)
  thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
  thresh = cv2.dilate(thresh, None, iterations=2)
  im2, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  temp=0
  for c in cnts:
    # if the contour is too small, ignore it
    if cv2.contourArea(c) > min_area:
      temp=1
  return temp   


if __name__ == '__main__':
    assure_path_exists("trainer/")#check if the trainer folder exists

    # Load the trained mode
    recognizer.read('trainer/trainer.yml')#load the training data into the recognizer
    
    # Read first frame.
    ok,frame = clickImage()
    while True:
        ok,frame = clickImage()#capture a frame
        frame_resized = frameResize(frame)
        frame_resized_grayscale = grayScaleConverter(frame_resized)
        min_area=(3000/800)*frame_resized.shape[1]# defining min cuoff area

        while True:
          previous_frame = frame_resized_grayscale
          grabbed, frame = clickImage()#capture a frame
          if not grabbed:
            break
          frame_resized = frameResize(frame)
          frame_resized_grayscale = grayScaleConverter(frame_resized)
          temp=background_subtraction(previous_frame, frame_resized_grayscale, min_area)#call fucntion background_subtraction
          if temp==1:   
            detect_all(frame_resized)#if there is a change in image then send for detection
        camera.release()
        cv2.destroyAllWindows()
