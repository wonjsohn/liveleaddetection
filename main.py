__author__ = 'wonjoonsohn'

# -*- coding: utf-8 -*-

# note: speed of the loop for processing taken video doesn't matter.
#       when taking video, I should get while loop as fast as I can...
#       Video recording can be fast without real time video feedback. Let's do that next.


############## Local library  ###########################
from snapshots import take_snapshot
from arguments import get_arguments
from shape_detection import detect_circle, detect_obstacles, detectShapesInTable
from save import save_output_at, save_video, save_dataframe
import constants as C
from check_camera_position import check_camera


################### openCV etc.##########################
import numpy as np
import pandas as pd
import pickle
import cv2
import imutils
from imutils.video import FileVideoStream
from imutils.video import WebcamVideoStream
from imutils.video import VideoStream
from imutils.video import FPS
from threading import Thread
from collections import deque
import sys
from scipy import misc
import glob
import os
import pdb
import time
import datetime
import csv
from multiprocessing import Process  # parallel process
import gc  # to disable garbage collection during appending data

############### popup GUI ############################
from PyQt5 import uic
from PyQt5 import QtCore, QtGui, QtWidgets
import pygame

##########################################################


####################################################
################  definitions ######################
####################################################
camera_port =0


# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
# http://www.rapidtables.com/web/color/RGB_Color.htm

# green filter is robust. 
#greenLower = (29, 86, 6) #  BGR  GREEN darker
greenLower = (24, 142, 0) #  BGR  GREEN darker ##temp at night
#greenUpper = (64, 255, 255) # BGR  GREEN lighter  %
greenUpper = (120, 255, 255)
# Yellow is too close to white board background and confused when ambient light is different. 
yellowLower = (16, 54, 40) # BGR  yellow postit.  
yellowUpper = (45, 255, 255)  # BGR  yellow postit

blueLower = (0, 211, 0) # BGR blue postit. 
blueUpper = (120, 255, 255)  # BGR  blue postit

orangeLower = (0, 141, 112) # BGR oramnge postit. 
orangeUpper = (243, 255, 255)  # BGR  orange postit

redLower = (44, 170, 114)
redUpper = (243, 219, 255)



""" for live lead migration detection (2024)"""
def small_lead_tracking(hsvmask):
    #########################################################
    ##### track minimum enclosing elipse of this mask #######
    #########################################################
    hsv = hsvmask
    hsvmask = cv2.inRange(hsv, orangeLower, orangeUpper)  # cheap
    # hsvmask3d = cv2.merge([zeros, zeros, hsvmask])
    kernel = np.ones((2, 2), np.uint8)
    img_erosion = cv2.erode(hsvmask, kernel, iterations=1)  #
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)  # erode/dial: 1.3ms
    # img_dilation3d = cv2.merge([zeros, img_dilation, zeros])

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL,  # 0.5ms
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]

    # Filter for small contours based on area
    min_contour_area = 5  # Adjust this value to fit the expected size of the small spot
    max_contour_area = 100  # Adjust to ignore large objects if any
    cnts = [c for c in cnts if min_contour_area < cv2.contourArea(c) < max_contour_area]

    center = None
    # only proceed if at least one contour was found
    # print "cnts", len(cnts)
    if len(cnts) > 0:
        # Get the first (largest) contour within the small area range
        largest_contour = cnts[0]
        # Compute moments to get the centroid of the contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = -1, -1  # If no contour found

        # Return the centroid of the small spot
        return cX, cY, len(cnts), img_dilation

    return -1, -1, 0, img_dilation  # If no small spot is detected


""" for live lead migration detection (2024)"""
def multi_small_lead_tracking(hsvmask):
    #########################################################
    ##### track minimum enclosing elipse of this mask #######
    #########################################################
    hsv = hsvmask
    hsvmask = cv2.inRange(hsv, orangeLower, orangeUpper)  # cheap
    # hsvmask3d = cv2.merge([zeros, zeros, hsvmask])
    kernel = np.ones((3, 3), np.uint8)
    img_erosion = cv2.erode(hsvmask, kernel, iterations=1)  #
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)  # erode/dial: 1.3ms
    # img_dilation3d = cv2.merge([zeros, img_dilation, zeros])

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL,  # 0.5ms
                            cv2.CHAIN_APPROX_SIMPLE)[-2]


    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]

    # Filter for small contours based on area
    min_contour_area = 10  # Adjust this value to fit the expected size of the small spot
    max_contour_area = 300  # Adjust to ignore large objects if any
    contours = [c for c in cnts if min_contour_area < cv2.contourArea(c) < max_contour_area]
    blobs = []

    for cnt in contours:
        # Get moments to find centroid
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            blobs.append((cx, cy, cnt))

    return blobs


####################################################
###                    main                     ####
####################################################
def run_main(timeTag, game_name):
    ## KEYBOARD command:  esc / "q" key to escape, "d" / "D" key to delete the trial.
    print("KEYBOARD command:  esc / q key to escape, d / D key to delete the trial.")

    # """ get screen resolution info"""
    pygame.init()
    screen_w = pygame.display.Info().current_w
    screen_h = pygame.display.Info().current_h
    pygame.quit() # check if this line needs to be disabled
    print(screen_w)
    print(screen_h)
    # screen_w = 1707  # just for now 0913
    # screen_h = 1067  # just for now 0913

    # global width, height, lens_h, cup_h
    #
    # width = 640
    # height = 480 #360
    lens_h = 0.8 #Lens vertical height 
    cup_h = 0.08 #Cuptop height 

    global running 
    running = True
    # color filter range for calcHist
    cfLower =greenLower
    cfUpper =greenUpper
    
    pts = deque(maxlen=args["buffer"])
    pts_orig = deque(maxlen=args["buffer"])

############  camera alignment ###############
    centerx = int(C.WIDTH/2)
    centery = int(C.HEIGHT/2)
    #  new table inner dimension: 87cm x 57cm.  (34x22)
    magfactor = 3
    table_width_in_cm = 87#20
    table_height_in_cm = 57#40
    table_halfw = table_width_in_cm*magfactor# (large), smaller: 230 table half-width approx. unit in px
    table_halfh = table_height_in_cm*magfactor# # table half-height approx.  # unit in px.

    ### Check if you need another snapshot during experiment due to disturbed camera or table.
    need_to_take_snapshot= False
    need_to_take_snapshot= check_camera(args, C.WIDTH, centerx, centery,table_halfw, table_halfh, timeTag, camera_port, screen_w,screen_h)

                
    """ take a snapshot of the board in png"""
    if need_to_take_snapshot:  #
        # img_name, circles, rectangles =take_snapshot(args, width, centerx, centery,table_halfw, table_halfh, timeTag, camera_port, screen_w,screen_h)
        img_name = take_snapshot(args, C.WIDTH, centerx, centery, table_halfw, table_halfh, timeTag,
                                                  camera_port, screen_w, screen_h)

    else: #if args.get("video", False):
        print("load snapshot, load pickle data")
        if args.get("video", False):  # postprocessing
            vname = args["video"][1]

            img_name = vname[:15]+".png"
    rectangles = None

    ### if the video path was not supplied, grab the reference to the camera
    if not args.get("video", False):
        print("no video ----------")
        if args.get("thread", False):
            print("thread  started #################################")
            cap = WebcamVideoStream(src=camera_port).start()
            # Set the desired resolution (1280x800)


        else:
            cap = cv2.VideoCapture(camera_port)
            # Set the resolution (e.g., 1280x720)
            # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
    # otherwise, load the video
    else:
        print("yes video ----------")
        if args.get("thread", False):
            cap = WebcamVideoStream(args["video"]).start()
        else:
            #videopath = "Output/" +args["video"][0] + "/videoOutput/"+  args["video"][1] #FIX: CROSS PLATFORM
            videopath = os.path.join("Output", "videoOutput",  args["video"][1]) #FIX: CROSS PLATFORM
            cap = cv2.VideoCapture(videopath)

    #cap.set(cv2.cv.CV_CAP_PROP_FPS, 60) #attemp to set fps recording (doesn't work)
    
    fps = FPS().start()
    time.sleep(1.0)
           
    # Read the first frame of the video
    if args.get("thread", False):
        frame = cap.read()
    else:
        ret, frame = cap.read()


    """ initial blob detection for multiple leads  (next 10 lines)"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # expensive 1.5ms
    hsv_mask, init_blobs = detect_initial_blobs(hsv)

    # hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # hsv_mask = cv2.inRange(hsv_frame, orangeLower, orangeUpper)
    roi_histograms = []

    # Initialize ROI histograms for each blob
    for window in init_blobs:
        x, y, w, h = window
        roi = hsv[y:y + h, x:x + w]

        # Use mask to focus on relevant pixels
        roi_mask = hsv_mask[y:y + h, x:x + w]

        roi_hist = cv2.calcHist([roi], [0], roi_mask, [180], [0, 180])
        roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        roi_histograms.append(roi_hist)


    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*args["codec"])

    writer = None
    writer_raw = None
    (h, w) = (None, None)
    zeros = None
                            
    # Cup data to write to file
    dataOut = []
    elapsedTimeList= []
    xObjectList=[]
    yObjectList=[]
    x1Object_List=[]
    x2Object_List = []
    lead1List = []
    lead2List = []
    start_cueList= []
    videoList=[]
    startTimeList = []
    reachTimeList = []
    goalReachedList = []
    # pandas dataframe output
    data = pd.DataFrame([])
    
    # Start time
    startTime = time.time()*1000.0
    startTimeRaw = time.time()
    startTimeFormatted = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3]
    """ converts to human readable: datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')"""

    num_frames = 0
    start_cue = 0
    delete_trial = 0 # whether to delete this trial because it is bad trial
    inTargetzone = 0  #  in P2P, you are in target zone.
    forceQuit = 0  # when you press 'c' to force quit at the end OR time is up.
    reach_time = 0
    targetzone_time = 0
    reach_time_start = 0
    marker = 0
    prevxObject = 0
    prevyObject = 0
    in_circle = 0
    ballEscaped = 0
    obstacleHit = 0


#########################################################
######## white virtual static frame #####################
#########################################################
    if args["virtual"] >0:
        whiteimage = np.zeros((C.HEIGHT, C.WIDTH, 3), np.uint8)
        whiteimage[:] = (255, 255, 255)
        # if args["tasktype"] == "p2p":

            # for approx in rectangles:
            #     cv2.drawContours(whiteimage, [approx], -1, (0, 128, 255), 3)
        # else: # fig8
        #     for (x, y, r) in circles:
        #         # draw the circle in the output image, then draw a rectangle
        #         # corresponding to the center of the circle
        #         cv2.circle(whiteimage, (x, y), r, (0, 255, 0), 4)
        #         cv2.rectangle(whiteimage, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)


        # draw lens registration target window 
        cv2.circle(whiteimage, (centerx, centery), 4, (0, 0, 255), 4)
        cv2.rectangle(whiteimage, (centerx- table_halfw, centery - table_halfh), (centerx + table_halfw, centery+ table_halfh), (153, 255, 255), 3)

        whiteimage_blank = whiteimage.copy()

    pathtype = "NONE"
    categories = "M_XX"


################################################
############### main loop  #####################
################################################
    x_pr = 0
    y_pr = 0
    vx_pr = 0
    vy_pr = 0
    bulk_trial_num = 0

    lefty = centery - table_halfh
    righty = centery + table_halfh
    leftx = centerx - table_halfw
    rightx = centerx + table_halfw

    cv2.namedWindow('main') # main window name
    """ Window positioning """
    cv2.moveWindow("main", 515, 0)  # Move it to (x, y)


    while running:
        if args.get("thread", False): # cheap
            frame_orig = cap.read()
            if not cap.grabbed: # fail to read a frame
                print("=====================None Frame ======================")
                continue # never catches anything
            frame_raw = frame_orig.copy()
        else:
            ret, frame_orig = cap.read()
            if not ret:
                break

        """ masking out unwanted region"""
        mask = np.zeros(frame_orig.shape, np.uint8)
        mask[lefty:righty, leftx:rightx] = frame_orig[lefty:righty, leftx:rightx]

        """ Resize the frame to increase speed."""
        frame = imutils.resize(mask, width=C.WIDTH)
        if not args.get("video", False):  # NOT a postprocessing mode
            frame_raw = imutils.resize(mask, width=C.WIDTH)

        """ Region of Interest (ROI) setting. Added 20181113
            to prevent detection out of the table."""
        #frame = frame_resized[centery-table_halfh:centery+table_halfh, centerx-table_halfw:centerx+table_halfw]
        #frame = frame_resized[0:640, 0:480]
            
        if args["virtual"] >0: # renew blank image (in this way, speeds of 3-4fps)
            whiteimage = whiteimage_blank.copy()

        t = time.time()

        # """ bulk recording (e.g. 15 trials in one run) not fully implemented 2018.12.12."""
        # if (args["bulkrecording"] > 0) & (bulk_trial_num >0):
        #     """ Bulk recording (like 15 trials (iw/ow) at once)
        #     produces files (dataframe + video) per trial.
        #     """
        #     timeTag = time.strftime("%Y%m%d_%H%M%S")  # get a new timetag
        #     writer = None #reset for making another video file

        if args.get("timed", False) > 0:  # time limit was set in -t argument.
            if args["tasktype"]=="p2p":
                if inTargetzone >0 and marker == 0: # end the trial when goal is reached.
                   reach_time = time.time()*1000 - startTime
                   print(reach_time)
                   targetzone_time = time.time()*1000 # the time when the cup entered the target
                   marker = 1

                                      
            if  start_cue < 1 and time.time()*1000.0 - startTime > 1*1000.0:  # start go sound at 1 second. 
                # GoSound.play(0)
                start_cue = 1
                
            ### Go! Text displayed 
            if start_cue == 1 and time.time()*1000.0 - startTime < 2.5*1000.0:
                cv2.putText(frame, "Go! ", (400, 40),
                            cv2.FONT_HERSHEY_TRIPLEX, 2.0, (224, 255, 255), 13) #
                if args["virtual"] >0:
                    cv2.putText(whiteimage, "Go! ", (400, 40),
                                cv2.FONT_HERSHEY_TRIPLEX, 2.0, (0, 0, 0), 13) # color black

            
            if  time.time()*1000.0 - startTime > args.get("timed", False) * 1000.0:  # in seconds (termination time)                           
                seconds = (time.time()*1000.0 - startTime)/1000.0
                forceQuit = 1
                # Calculate frames per second
                fps_calc  = num_frames / seconds;
                print("Time taken: ", args.get("timed", False), "s, fps: {0}".format(fps_calc))
                break

   
        # check if the writer is None
        if writer is None:
            # store the image dimensions, initialzie the video writer,
            # and construct the zeros array
            if args.get("video", False):  # postprocessing mode
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:
                width = C.WIDTH    # 640
                height = C.HEIGHT  # 360 (480 doesn't write to file)
                
            videoName_path = save_video(args, timeTag, categories, gameTest, isRaw=0)

            print('fourcc:', fourcc)
            print('w, h:', width, height)
            writer = cv2.VideoWriter(videoName_path, fourcc, args["fps"],
                                     (width, height), True)
            zeros = np.zeros((height, width), dtype="uint8")

            """ Raw video recording for back up (slows down fps)"""
            if args["rawvideo"] > 0:
                rawvideoName_path = save_video(args, timeTag, categories, gameTest, isRaw=1)
                writer_raw = cv2.VideoWriter(rawvideoName_path, fourcc, args["fps"],
                                             (width, height), True)

	# blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # expensive 1.5ms


########  Choose the type of tracking algorithm. ########################
### "cl_object"- minimum enclosing circle
### "cl_object+kalman" - minimum enclosing circle + kalman filter
### "el_object" - minimum enclosing elipse. (better when cup is warped toward the edge)
########################################################################
        global xObject, yObject    # have to check if making these global slows down.  WJS.

        if gameTest:
            xObject, yObject = (10, 10)  # camera is not used. trackpad / mouse cursor is used.
            cupx = 100
            cupy = 100
        else:
            if args["marker"] == "camshift_multi_object":  #
                # Iterate through each blob and update tracking with CamShift
                for i, window in enumerate(init_blobs):
                    print(i)
                    print(window)
                    # Create backprojection
                    back_proj = cv2.calcBackProject([hsv], [0], roi_histograms[i], [0, 180], 1)

                    # # Apply threshold to reduce noise in the backprojection
                    # _, back_proj = cv2.threshold(back_proj, 50, 255, cv2.THRESH_BINARY)

                    # Apply CamShift to get new position of the blob
                    ret, new_window = cv2.CamShift(back_proj, window,
                                                   (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))
                    print(new_window)
                    # Draw the tracking window
                    points = cv2.boxPoints(ret)
                    points = np.int0(points)
                    cv2.polylines(frame, [points], True, (0, 255, 0), 2)

                    # Update the window for next iteration
                    init_blobs[i] = new_window

                # TODO Change
                xObject = 0
                yObject = 0
                cupx = xObject
                cupy = yObject


            elif args["marker"] == "multi_small_blob_object":  # track ball
                blbs = multi_small_lead_tracking(hsv)
                cxx = []
                cyy = []
                print(blbs)
                blbs_ordered = reorder_points(blbs)

                for i, (cx, cy, cnt) in enumerate(blbs_ordered):
                    print(str(i) + " " + str(cx) + " "+ str(cy))
                    cxx.append(cx)
                    cyy.append(cy)
                    # # Draw the center of the blob
                    # cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # Green center
                    # # Draw contour of the blob
                    # # cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 2)  # Yellow contour
                    # # Label the blob
                    #
                    cv2.putText(frame, f'Blob {i + 1}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                """ reorder points to our rule """
                """ 3   2 """
                """ 0   1  """
                lead_count = 6 # for octrode, assuming we don't use the last two end leads.
                lead1_positions= calculate_inbetween(cxx[3], cyy[3], cxx[0], cyy[0], lead_count)
                lead2_positions= calculate_inbetween(cxx[2], cyy[2],cxx[1], cyy[1], lead_count)
                for i, (cx, cy) in enumerate(lead1_positions):
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)  # Green center
                for i, (cx, cy) in enumerate(lead2_positions):
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)  # Green cente2

                # TODO Change
                xObject = 0   #dummy
                yObject = 0    #dummy
                cupx =  xObject #dummy
                cupy =   yObject #dummy
                x1Object_list =lead1_positions
                x2Object_list =lead2_positions

            elif args["marker"] == "small_blob_object+kalman": # track ball

                (xObject, yObject, len_cnts, img_dilation) = small_lead_tracking(hsv)

                ### Jumping prevention (temporary)
                threshold = 60
                #xObject, yObject = prevent_jumping(num_frames, xObject, prevxObject, yObject, prevyObject, threshold)

                measured = np.array((xObject, yObject), np.float32)

                # use to correct kalman filter
                kalman.correct(measured);  # input: measurement

                # get new kalman filter prediction
                prediction = kalman.predict();  # Computes a predicted state.

                if len_cnts > 0:
                    # only proceed if the radius meets a minimum size
                    if args["trace"] > 0:
                        # if radius > 5:
                        cv2.circle(frame,  (int(prediction[0]), int(prediction[1])), 2, (0, 0, 255), -1)
                cupx=xObject
                cupy=yObject

            elif args["marker"] == "small_blob_object": # track ellipse ball
                (xObject, yObject, len_cnts, img_dilation) = small_lead_tracking(hsv)

                if len_cnts > 0:
                    # only proceed if the radius meets a minimum size
                    if args["trace"] > 0:
                        cv2.circle(frame, (int(xObject),int(yObject)), 3, (0, 0, 255), -1)
                cupx = xObject
                cupy = yObject

            # previous frame's coordinates
        prevxObject = xObject
        prevyObject = yObject




################# virtual display test ###############
        if args["virtual"] >0:
            cv2.circle(whiteimage,(int(cupx),int(cupy)), 30, (0, 128, 255), 2)
            cv2.circle(whiteimage, (int(cupx),int(cupy)), 5, (0, 0, 255), -1)


        currentTime = time.time()*1000
        elapsedTime = currentTime- startTime

        ## increase performance than appending to dataframe 
        elapsedTimeList.append(elapsedTime)
        start_cueList.append(start_cue)
        videoList.append(args["video"])
        startTimeList.append(startTimeRaw)
        reachTimeList.append(reach_time)

        xObjectList.append(xObject)
        yObjectList.append(yObject)

        lead1List.append(x1Object_list)
        lead2List.append(x2Object_list)

        pts.appendleft((int(cupx), int(cupy)))

        ### Drawing clock, line trace, and traces. (even if display=0, it can be written to file)
        if args["clock"] > 0:
            drawClock(frame, num_frames, elapsedTime,  timeTag, virtual=0)
            # if not args.get("video", False):  # NOT a postprocessing mode
            #     drawClock(frame_raw, num_frames, elapsedTime, timeTag, virtual=0)
            if args["virtual"] >0:
                drawClock(whiteimage, num_frames, elapsedTime,  timeTag, virtual=1)
           
        if len(pts) > 100:
            pts_draw = pts[:100] # keep last 100 points to draw line.
        else:
            pts_draw = pts

        ### line trace
        if args["linetrace"] > 0:
            # loop over the set of tracked points (improve to be < O(n))
            for i in range(1, len(pts_draw)):  # what is xrange in python2 is range in python 3
                    # if either of the tracked points are None, ignore
                    # them
                    if pts_draw[i - 1] is None or pts_draw[i] is None:
                            continue
     
                    # otherwise, compute the thickness of the line and
                    # draw the connecting lines
                    thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)

                    ## line trace drawing is expensive
                    if args["virtual"]>0:
                        cv2.line(whiteimage, pts_draw[i - 1], pts_draw[i], (255, 0, 0), thickness)#blue
                

        """"For demo video recording (4 display windows in a frame)"""
        #img_dilation2 = np.dstack([img_dilation] * 3)
        #output =  np.vstack((np.hstack([frame_raw, hsv]), np.hstack([img_dilation2, frame])))

        """ display"""
            # cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        if args["display"] > 0:
            cv2.imshow("main", frame)  # expensive
            # cv2.imshow("Frame", frame)  # expensive
        if args["virtual"] > 0:
            cv2.imshow("virtualFrame", whiteimage)

        ###########  KEYBOARD INPUTS (typical) ##############

	# if the 'q' key is pressed, stop the loop
        k= cv2.waitKey(1) & 0xFF   # very expensive. 19ms. # default output = 255 (at least in python 3.6)

        if k == 27: # esc (Break and save)
            goalReachedList.append(0)
            break
        elif k == 67 or k == 99: # "C" and "c" key: completed the fig8 task.
            inTargetzone = 1
            goalReachedList.append(inTargetzone)
            break
        elif k == 68 or k==100: # "D" and "d" key: delete.
            delete_trial = 1
            break
        goalReachedList.append(inTargetzone)

        # write the frame
        writer.write(frame)   # 1.3ms
        if args["rawvideo"] > 0:
            writer_raw.write(frame_raw)  # record raw for postprocessing see how much it slows down.
        #writer.write(whiteimage)   # 1.3ms   # TODO: make it automatic selection

        prev_num_frames = num_frames
        num_frames = num_frames + 1
        fps.update()  # update the FPS counter
        
    ### stop the timer and display FPS information
    fps.stop()
    
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    print(len(elapsedTimeList), len(xObjectList), len(yObjectList))  # for debugging
    print(len(lead1List), len(lead2List)) # for debugging
    print(len(start_cueList), len(videoList), len(reachTimeList), len(goalReachedList)) # for debugging

    ## TODO: simplfy 
    if not delete_trial:
        if args["marker"] == "multi_small_blob_object" or args["marker"] == "camshift_multi_object" or args["marker"] == "cl_object+kalman": # track ball
            """ with meta data written on top """
            data = pd.DataFrame(
                {'elapsedTime': elapsedTimeList, 'lead1_List': lead1List, 'lead2_List': lead2List,
                 'videoReplay': videoList})

        """ GUI popup to ask if the trial was a success"""
        # x = int(input("Was this trial a success?  Y(1) or N(0) "))
        # replace with GUI to ask if this was a successful trial.
        import graphical_panel.popup_window as POPUP
        app = QtWidgets.QApplication(sys.argv)
        w = POPUP.Window()
        # w.setWindowTitle('User Input')
        # w.show()
        retval = [None] * 3
        i = 0
        for ch in w.get_data():
            retval[i] = ch
            i = i+1
        isSucess = retval[0]
        note = retval[2]


        dir_of_move = 'any'

        sharedFileName = save_dataframe(data, isSucess, args, timeTag, categories, gameTest, startTimeFormatted, note, dir_of_move, ballEscaped, obstacleHit, pathtype)  # write dataframe to file
        writer.release()
        if args["rawvideo"] > 0:
            writer_raw.release()

        """ this is to make the video file name same as data frame (csv) file name (to indicate success trials)"""
        if not args.get("video", False):  # if not video mode
            local_path = os.getcwd()
            print('local path:', local_path)
            dataOutput_path = os.path.join(str(local_path), "Output", "videoOutput")
            print('dataOutput_path:', dataOutput_path)
            print('sharedFileName:', sharedFileName)
            newVideoName_path = os.path.join(dataOutput_path, sharedFileName+ ".mp4")
            print('newVideoName_path:', newVideoName_path)
    else:
        writer.release()
        if args["rawvideo"] > 0:
            writer_raw.release()
        print(videoName_path)
        os.remove(videoName_path) # delete this trial

    if not args['thread'] >0:
        cap.release()
    cv2.destroyAllWindows()


# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 1 )

def prevent_jumping(num_frames, xObject, prevxObject, yObject, prevyObject, threshold):
    ####################################################
    #### prevent detection jumping when more than ######
    #### one object w/ same color exists          ######
    ####################################################
    if num_frames > 0 and ((abs(xObject - prevxObject) > threshold ) or (abs(yObject - prevyObject) > threshold)):#e.g. threshold# pixel jump compared to previous frame
        xObject = prevxObject
        xObject = prevyObject
    else:
        xObject = xObject
        yObject = yObject
    return xObject, yObject

def kalman_filter_init():
    ####################################################
    ####    Initialize Kalman filter           #########
    ####################################################
    # init kalman filter object
    global kalman, measurement, prediction
    
    kalman = cv2.KalmanFilter(4,2)
    kalman.measurementMatrix = np.array([[1,0,0,0],
                                         [0,1,0,0]],np.float32)

    kalman.transitionMatrix = np.array([[1,0,1,0],
                                        [0,1,0,1],
                                        [0,0,1,0],
                                        [0,0,0,1]],np.float32)

    kalman.processNoiseCov = np.array([[1,0,0,0],
                                       [0,1,0,0],
                                       [0,0,1,0],
                                       [0,0,0,1]],np.float32) *0.0000001  # Tune this parameter.

    measurement = np.array((2,1), np.float32)
    prediction = np.zeros((2,1), np.float32)

    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,1 ) # not used?


"""Function to detect blobs using color segmentation  (0913)"""
def detect_initial_blobs(hsvmask):
    #########################################################
    ##### track minimum enclosing elipse of this mask #######
    #########################################################
    hsv = hsvmask
    hsvmask = cv2.inRange(hsv, orangeLower, orangeUpper)  # cheap
    # hsvmask3d = cv2.merge([zeros, zeros, hsvmask])
    kernel = np.ones((2, 2), np.uint8)
    img_erosion = cv2.erode(hsvmask, kernel, iterations=1)  #
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)  # erode/dial: 1.3ms
    # img_dilation3d = cv2.merge([zeros, img_dilation, zeros])

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL,  # 0.5ms
                            cv2.CHAIN_APPROX_SIMPLE)[-2]

    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]

    # Filter for small contours based on area
    min_contour_area = 4  # Adjust this value to fit the expected size of the small spot
    max_contour_area = 200  # Adjust to ignore large objects if any
    contours = [c for c in cnts if min_contour_area < cv2.contourArea(c) < max_contour_area]
    first_blobs = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # Get bounding box around blob
        first_blobs.append((x, y, w, h))

    return hsvmask, first_blobs

def drawClock(frame, num_frames, elapsedTime, timeTag, virtual):
    ####################################################
    ### display date/fm on screen (run when recording)##
    ####################################################
    import datetime
    # draw the text and timestamp on the frame
    cv2.putText(frame, "frames: "+str(num_frames), (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (224, 255, 255), 1) # color black
    cv2.putText(frame, "Stopwatch: "+str('%0.3f' %(elapsedTime/1000))+"s", (150, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (224, 255, 255), 1) # color black
    if virtual:
        cv2.putText(frame, "frames: " + str(num_frames), (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # color black
        cv2.putText(frame, "Stopwatch: " + str('%0.3f' % (elapsedTime / 1000)) + "s", (150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # color black
    #cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S.%f%p")[:-5],
#                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 153), 1) # color black



def calculate_lead_positions_between(xa, xb, ya, yb, lead_count):
    list_of_leads = []
    """  3   2  """
    """" 0   1  """
    """  but I cannot assume this!"""
    """ assumption xa = 3"""

    for x in (n + 1 for n in range(lead_count-2)):
        list_of_leads.append((min(xa, xb) +  int((xa - xb) / lead_count *x ), int(min(ya, yb) +  (ya - yb) / lead_count * x)))

    return list_of_leads


def calculate_inbetween(x1, y1, x2, y2, n):
    # Calculate the step size for x and y coordinates
    x_step = (x2 - x1) / (n + 1)
    y_step = (y2 - y1) / (n + 1)

    # Generate the n intermediate points
    points = []
    for i in range(1, n + 1):
        new_x = int(x1 + i * x_step)
        new_y = int(y1 + i * y_step)
        points.append((new_x, new_y))

    return points

# Sample code to find contours and then reorder
def reorder_points(pts):
    # Convert the list of points to a NumPy array
    pts = np.array(pts)

    # Sort based on the y-coordinate (top to bottom)
    pts = sorted(pts, key=lambda x: x[1])

    # Now separate the points into the top and bottom
    top_points = sorted(pts[:2], key=lambda x: x[0])   # Sort by x-coordinate (left to right) for top
    bottom_points = sorted(pts[2:], key=lambda x: x[0])  # Sort by x-coordinate (left to right) for bottom

    # Reorder the points in the desired order:
    # lower-left, lower-right, upper-right, upper-left
    return np.array([bottom_points[0], bottom_points[1], top_points[1], top_points[0]])

if __name__ == "__main__":
    from scipy.io import savemat, loadmat
    timeTag = time.strftime("%Y%m%d_%H%M%S")

    import argparse
    import math, random, sys

    ###  get arguments from terminal
    global args  # check speed performance...
    args = get_arguments()
    #
    global gameTest
    gameTest = args["gametest"]
    # game_name = 'traffic_dodger';  ##  OPTIONS: 'banana_game', 'traffic_dodger'
    game_name = 'dummy'

    #########################################################
    if args["mode"] =="realtime_plot":
        import sys
        import graphical_panel.realtime_magictable as MONITOR_WINDOW
        global win, p
        win = MONITOR_WINDOW.magictable_monitor_window()
        # p = Process(target=MONITOR_WINDOW.magictable_monitor_window(), args=())
        # sys.exit(win.app.exec_())

    kalman_filter_init()

    run_main(timeTag, game_name) # Main openCV loop
    
