# USAGE
# python fish_tracking.py --video videos/12.avi --tracker csrt

# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
from pyimagesearch.centroidtracker import CentroidTracker
import numpy as np
firstFrame = None
secondFrame = None
frame = None
top_ff=None
bottom_ff=None

#rects=[]


def read_frame(frame):
	#frame = vs.read()
	#frame = frame[1] if args.get("video", False) else frame
	## check to see if we have reached the end of the stream
	#if frame is None:
	#	break
	#return
	f=vs.read()
	f = f if args.get('video', None) is None else f[1]
	#print(f)
	return f

def modif_frame(f):
	#f = imutils.resize(f, width=500)
	grayf = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
	#grayf = cv2.GaussianBlur(grayf, (21, 21), 0)
	return grayf

def remove_duplicates(x):
  return list(dict.fromkeys(x))

def Detector(grayA,grayB,rects):
	global frame#,diff_frame,thresh_frame
	diff_frame=None
	thresh_frame=None
	rects.clear()
	diff_frame = cv2.absdiff(grayA, grayB)
	thresh_frame = cv2.threshold(diff_frame, 5, 255, cv2.THRESH_BINARY)[1] 
	thresh_frame = cv2.dilate(thresh_frame, None, iterations = 1)
	
	# Finding contour of moving object 
	cnts,_ = cv2.findContours(thresh_frame.copy(),  
					cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
	#change here so bounding boxes dont affect frame
	
	contours_poly = [None]*len(cnts)
	boundRect = [None]*len(cnts)
	for i, contour in enumerate(cnts): 
		if cv2.contourArea(contour) > 100: 
			contours_poly[i] = cv2.approxPolyDP(contour, 3, True)
			boundRect[i] = cv2.boundingRect(contours_poly[i])
			rects.append((int(boundRect[i][0]), int(boundRect[i][1]),int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])))
			cv2.rectangle(thresh_frame, (int(boundRect[i][0]), int(boundRect[i][1])),(int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (0, 0, 255), 2)

	return rects,diff_frame,thresh_frame

def Parse(obsA,obsB):
	obs=[]
	obs_=[] 
	flag=0
	for (objectID, centroid) in obsA.items():
		for (ID,cent) in obsB.items():
			if abs(centroid[1]-cent[1])<15 and (not obs or objectID != any(sl[0] for sl in obs)): 
				obs.append([objectID, centroid[0],int((centroid[1]+cent[1])/2),cent[0]])

	
	
	for subl in obs:
		flag=0
		if (not obs_): 
			obs_.append(subl)
		else:
			for sl in obs_:
				if(subl[0] == sl[0]):
					flag=1
			if(flag==0):
				obs_.append(subl)
	
	return obs_

def CTracker(p,frame,rects):
	if(p==0):
		objects = ct1.update(rects)
	else:
		objects = ct2.update(rects)
	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 2, (0, 255, 0), -1)
		print(centroid[0], centroid[1])
	return frame

def Tracker(frame,obs,p):
	for (objectID, x,y,z) in obs:
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
		if(p==1):
			cv2.putText(frame, text, (x - 10, y - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
			print(x,y)
		else:
			cv2.putText(frame, text, (z - 10, y - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(frame, (z, y), 2, (0, 255, 0), -1)
			print(z,y)
	return frame

ct1 = CentroidTracker(5)
ct2 = CentroidTracker(5)
(H, W) = (None, None)
end=False
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

# initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()


# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
	#print("fish video")

# loop over frames from the video stream
while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	
	#rects = []
	rectsA=[]
	rectsB=[]
	obs1=[]
	obs2=[]
	out=[]
		
	if firstFrame is None:
		firstFrame=read_frame(firstFrame)
		firstFrame = imutils.resize(firstFrame, width=400)
		h, w, channels = firstFrame.shape
		print(h/2,w)
		#firstFrame = firstFrame[0:h,60:w-90] #60
		#grayA=modif_frame(firstFrame)
		top_ff=firstFrame[30:int(h/2)-70,60:w-90] #210,250
		top_grayA=modif_frame(top_ff)
		bottom_ff=firstFrame[int(h/2)+50:h-40,60:w-90] #210,250
		bottom_grayA=modif_frame( bottom_ff)

		
		#print(firstFrame)

	secondFrame = read_frame(secondFrame)
	# resize the frame (so we can process it faster)
	secondFrame = imutils.resize(secondFrame, width=400)
	#secondFrame = secondFrame[0:h,60:w-90] #60
	#grayB=modif_frame(secondFrame)
	top_sf=secondFrame[30:int(h/2)-70,60:w-90] #60
	top_grayB=modif_frame(top_sf)
	bottom_sf=secondFrame[int(h/2)+50:h-40,60:w-90] #60
	bottom_grayB=modif_frame( bottom_sf)

	rectsA,diffA,threshA=Detector(top_grayA, top_grayB,rectsA)
	rectsB,diffB,threshB=Detector(bottom_grayA, bottom_grayB,rectsB)

	for a in rectsA:
		cv2.rectangle(threshA, (a[0],a[1]),(a[2],a[3]), (0, 0, 255), 2)
	for b in rectsB:
		cv2.rectangle(threshB, (b[0],b[1]),(b[2],b[3]), (0, 0, 255), 2)

	#diff_frame = cv2.absdiff(grayA, grayB)
	#thresh_frame = cv2.threshold(diff_frame, 5, 255, cv2.THRESH_BINARY)[1] 
	#thresh_frame = cv2.dilate(thresh_frame, None, iterations = 1) 

	#top_diff = cv2.absdiff(top_grayA, top_grayB)
	#bottom_diff = cv2.absdiff(bottom_grayA, bottom_grayB)
	# Finding contour of moving object 
	#cnts,_ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
	#bbs=Detector(grayA,grayB)
	#print(bbs)
	#bbs.clear()
	frame=firstFrame
	firstFrame=secondFrame
	#grayA=grayB
	top_grayA=top_grayB
	bottom_grayA=bottom_grayB

	#for single frame
	#contours_poly = [None]*len(cnts)
	#boundRect = [None]*len(cnts)
	#for i, contour in enumerate(cnts): 
	#	if cv2.contourArea(contour) > 100: 
	#		contours_poly[i] = cv2.approxPolyDP(contour, 3, True)
	#		boundRect[i] = cv2.boundingRect(contours_poly[i])
	#		rects.append((int(boundRect[i][0]), int(boundRect[i][1]),int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])))
	
	#frameA=CTracker(0,top_sf,rectsA)
	print("FrameA")
	obs1 = ct1.update(rectsA)
	print(obs1)
	#frameB=CTracker(1,bottom_sf,rectsB)
	print("FrameB")
	obs2 = ct2.update(rectsB)
	print(obs2)
	if (obs1):
		out=Parse(obs1,obs2)	
	if (out):
		print(out)
		print("FrameA-avg")
		frameA=Tracker(top_sf,out,1)
		print("FrameB-avg")
		frameB=Tracker(bottom_sf,out,2)

	# show the output frame
		cv2.imshow("FrameA", frameA)
		cv2.imshow("FrameB", frameB)
	
	cv2.imshow("Threshold Frame A", threshA) 
	# Displaying the black and white image in which if 
	# intensity difference greater than 30 it will appear white 
	cv2.imshow("Threshold Frame B", threshB) 
	time.sleep(0.2)
	key = cv2.waitKey(1) & 0xFF

	#ignore the below code, potential alternate method still being explored

	# if the 's' key is selected, we are going to "select" a bounding
	# box to track
	#end=False
	if key == ord("s") :#or end==False
		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
		#boxes = cv2.selectROIs("Frame", frame, fromCenter=False,
		#	showCrosshair=False)
		#for i in boxes:
		#	(H,W)=(boxes[i][3]-boxes[i][1],boxes[i][3]-boxes[i][0])
		#	print(H,W)
		# create a new object tracker for the bounding box and add it
		# to our multi-object tracker
		#tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
		#trackers.add(tracker, frame, box)
		#while(end==False):
		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
		box = cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=False)
		#(H,W)=(box[1]-box[3],box[0]-box[3])
		#print(H,W)
		# create a new object tracker for the bounding box and add it
		# to our multi-object tracker
		tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
		trackers.add(tracker, frame, box)
		txt=input("Do you want to add more? Press d to stop ")
		if txt == "d":
			end=True
		else:
			continue
		
	if key == ord("p"):
		while(end==False):
			txt=input("Quit? Press q")
			if txt == "q":
				end=True
			else:
				continue
	#if key == ord("d"):
	#	end=True
	# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break

# if we are using a webcam, release the pointer
if not args.get("video", False):
	vs.stop()

# otherwise, release the file pointer
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()