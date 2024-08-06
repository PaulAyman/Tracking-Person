'''
1. read frame
2. setup ROI
3. tracker type + Init

LOOP-------------------------------
4. read frame
5. tracker update frame
6. if success -> draw bbox
7. imshow + wait
-----------------------------------
'''

import cv2
import imutils

TrDict = {
    'csrt': cv2.legacy.TrackerCSRT_create,
    'kcf' : cv2.legacy.TrackerKCF_create,
    'boosting' : cv2.legacy.TrackerBoosting_create,
    'mil': cv2.legacy.TrackerMIL_create,
    'tld': cv2.legacy.TrackerTLD_create,
    'medianflow': cv2.legacy.TrackerMedianFlow_create,
    'mosse': cv2.legacy.TrackerMOSSE_create
}

tracker = TrDict['csrt']()
#tracker = cv2.TrackerCSRT_create()

#v = cv2.VideoCapture(r'D:\mot.mp4') # video
v = cv2.VideoCapture("D:\Programing languages\Python\Tracking by Mouse\Road traffic video for object detection and tracking.mp4")

ret, frame = v.read()
frame = imutils.resize(frame,width=600)
cv2.imshow('Frame',frame)
bb = cv2.selectROI('Frame',frame)
tracker.init(frame,bb)

while True:
    ret, frame = v.read()
    if not ret:
        break
    frame = imutils.resize(frame,width=600)
    (success,box) = tracker.update(frame)   
    if success:
        (x,y,w,h) = [int(a) for a in box]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,255,0),2)
    cv2.imshow('Frame',frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
v.release()
cv2.destroyAllWindows()
        
    
    