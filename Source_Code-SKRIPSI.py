# USAGE
# python pi_face_recognition.py --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle
# import the necessary packages

from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import serial
import os, time
from datetime import datetime
import cv2
import RPi.GPIO as GPIO
import sys

sys.path.append('/home/pi/MFRC522-python')
#from mfrc522 import SimpleMFRC522
from mfrc522 import MFRC522

MIFAREReader = MFRC522()
#reader = SimpleMFRC522()
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
    help = "path to where the face cascade resides")
ap.add_argument("-e", "--encodings", required=True,
    help="path to serialized db of facial encodings")
args = vars(ap.parse_args())

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(args["encodings"], "rb").read())
detector = cv2.CascadeClassifier(args["cascade"])

"""def selenoid():
  
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)    
    GPIO.setup(18,GPIO.OUT)
    GPIO.output(18,GPIO.HIGH)
    time.sleep(5)
    GPIO.output(18,GPIO.LOW)
    GPIO.cleanup()"""
    
def selenoidOn():
  
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)    
    GPIO.setup(18,GPIO.OUT)
    GPIO.output(18,GPIO.HIGH)
    
def selenoidOff():
  
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)    
    GPIO.setup(18,GPIO.OUT)
    GPIO.output(18,GPIO.LOW)
    GPIO.cleanup()

def buzzer():
  
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)    
    GPIO.setup(16,GPIO.OUT)
    GPIO.output(16,GPIO.HIGH)
    time.sleep(1)
    GPIO.output(16,GPIO.LOW)
    GPIO.cleanup()

def buzzerOn():
  
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)    
    GPIO.setup(16,GPIO.OUT)
    GPIO.output(16,GPIO.HIGH)    
    
def buzzerOff():
  
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)    
    GPIO.setup(16,GPIO.OUT)
    GPIO.output(16,GPIO.LOW)
    GPIO.cleanup()

def cvtToString(s):
    list = [str(i) for i in s]
    res = int("".join(list))
    return (str(res))

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
buzzerOff()
# start the FPS counter
fps = FPS().start()

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to 500px (to speedup processing)
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    
    # convert the input frame from (1) BGR to grayscale (for face
    # detection) and (2) from BGR to RGB (for face recognition)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    # OpenCV returns bounding box coordinates in (x, y, w, h) order
    # but we need them in (top, right, bottom, left) order, so we
    # need to do a bit of reordering
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    # compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"],
            encoding)
        name = "Unknown"
       
        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
                                
                    #Scan for cards
                (status,TagType) = MIFAREReader.MFRC522_Request(MIFAREReader.PICC_REQIDL)
                    
                    #if a card is found
                if status == MIFAREReader.MI_OK:
                        print ("Card Detected")
                        
                    #Get the UID of the card
                (status,uid) = MIFAREReader.MFRC522_Anticoll()
                    
                    #if we have UID, continue
                if status == MIFAREReader.MI_OK:
                        
                        #print UID
                    print("Card read UID: "+str(uid[0])+","+str(uid[1])+","+str(uid[2])+","+str(uid[3])+","+str(uid[4]))
                        #This is the default key for authentication
                    key = [0xFF,0xFF,0xFF,0xFF,0xFF,0xFF]
                    
                        #select the scanned tag
                    MIFAREReader.MFRC522_SelectTag(uid)
                    
                        #Enter Your Card UID here
                    my_uid = [136,4,109,73,168]
                    my_uid1 = [136,4,66,44,226]
                                                 
                        #check to see if card UID read matches your card UID
                    if (uid == my_uid and name == "Dito") or (uid == my_uid1 and name == "Gigih") or ():
                        print(name)
                        print("Access Granted")
                        print("Door Opened")
                        selenoidOn()
                        print("selenoidON")
                        buzzerOn()
                        time.sleep(1)
                        
                        #SendingSMS()
                        GPIO.setmode(GPIO.BOARD)
                        port = serial.Serial("/dev/serial0", baudrate=9600, timeout=1)
                        time.sleep(1)
                        port.write(b'AT\r')
                        rcv = port.read(10)
                        print(rcv)
                        time.sleep(1)

                        port.write(b'AT+CMGF=1\r')
                        print("Text Mode Enabled")
                        time.sleep(1)
                        port.write(b'AT+CMGS="+6285776169745"\r')
                        now = datetime.now()
                        timetoday = now.strftime("%d/%m/%Y %H:%M:%S")
                        msg = "Door Opened.\n" "This e-KTP with ID Number: " + cvtToString(uid) + " Name: " + name + " Time: " + timetoday
                        print("sending message")
                        time.sleep(1)
                        port.reset_output_buffer()
                        time.sleep(1)
                        port.write(str.encode(msg+chr(26)))
                        time.sleep(1)
                        print("message sent")                    
                    
                    else:
                        print("e-KTP not matched")
                        print("Access Denied, Please Try Again")
                        buzzer()
                        time.sleep(1)
                        buzzer()
                        break
                                      
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
                GPIO.setwarnings(False)
                GPIO.setmode(GPIO.BOARD)
                pin = 7
                GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                input_state = GPIO.input(pin)
                if input_state == False :
                    selenoidOff()
                    print("selenoidOFF")
                    buzzerOff()
                    print("Door Closed")
                    
                    #SendingSMS()
                    GPIO.setmode(GPIO.BOARD)
                    port = serial.Serial("/dev/serial0", baudrate=9600, timeout=1)
                    time.sleep(1)
                    port.write(b'AT\r')
                    rcv = port.read(10)
                    print(rcv)
                    time.sleep(1)

                    port.write(b'AT+CMGF=1\r')
                    print("Text Mode Enabled")
                    time.sleep(1)
                    port.write(b'AT+CMGS="+6285776169745"\r')
                    now = datetime.now()
                    timetoday = now.strftime("%d/%m/%Y %H:%M:%S")
                    msg = "Door Closed.\n" "Name: " + name + " Time: " + timetoday
                    print("sending message")
                    time.sleep(1)
                    port.reset_output_buffer()
                    time.sleep(1)
                    port.write(str.encode(msg+chr(26)))
                    time.sleep(1)
                    print("message sent")
                                   
            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)
                
        elif False in matches:
            name = "Unknown"
            #counts[name] = counts.get(name, 0) + 1
            print("Face not matched")
            print("Access Denied, Please Try Again")
            buzzer()
            time.sleep(1)
            buzzer()            
            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            #name = max(counts, key=counts.get)
        
        # update the list of names
        names.append(name)
 
    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom),
            (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 255, 0), 2)

    # display the image to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()