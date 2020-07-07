import cv2
import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


vid_cam = cv2.VideoCapture(0)# Start capturing video 
# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_id = 7# For each person, one face id
count=107# Initialize sample face image
assure_path_exists("dataset/")

# Start looping.
while(True):
    _, image_frame = vid_cam.read()# Capture video frame
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)# Convert frame to grayscale
    # Detect frames of different sizes, list of faces rectangles
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    # Loops for each faces
    for (x,y,w,h) in faces:
        # Crop the image frame into rectangle
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1# Increment sample face image
        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        # Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('frame', image_frame)
    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # If image taken reach 100, stop taking video
    elif count>1000:
        break
vid_cam.release()# Stop video
cv2.destroyAllWindows()# Close all started windows
