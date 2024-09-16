import os


####################################################
### save at subdirectory, makedir if not exists ####
####################################################
def save_output_at(subdir):
    local_path = os.getcwd()

    dataOutput_path = os.path.join(str(local_path), "Output", subdir)
    if not os.path.exists(dataOutput_path):
        os.makedirs(dataOutput_path)
    return dataOutput_path


####################################################
### save video                                  ####
#################################################### 
def save_video(args, timeTag, categories, gameTest, isRaw):
    if gameTest:
        controller = "Mouse"
    else:
        controller = "Tracking"
    #timeTag = time.strftime("%Y%m%d_%H%M%S")

    if args.get("video", False):
        #dataOutput_path = save_output_at("videoOuput_PP")
        subjectOnly= os.path.splitext(args["video"][0])[0]
        local_path = os.getcwd()
        dataOutput_path2 = os.path.join(str(local_path), "Output", subjectOnly, "videoOutput_PP")
        if not os.path.exists(dataOutput_path2):
            os.makedirs(dataOutput_path2)

        filenameOnly= os.path.splitext(args["video"][1])[0]
        videoName_path = os.path.join(dataOutput_path2 , filenameOnly + "_PP.mp4") # PP: postprocessed # FIX:CROSS PLATFORM

    else:
        dataOutput_path = save_output_at("videoOutput")
        videoName_common = timeTag + "_" + args.get("condition") + "_" + args[
            'idlevel'] + "_" + categories + "_" + args["subject"] + "_" + controller + "_" + str(args["timed"]) + "s_" + str(
            args["fps"]) + "fps"
        if args["practice_boardtask"]: # if this is a practice trial
            videoName_common = videoName_common + "_PR"

        if isRaw:
            # videoName_path = os.path.join(dataOutput_path, timeTag + "_" + args.get("condition")  + "_" + args['idlevel'] + "_"+ categories + "_" + subjectID + "_" + controller + "_" + str(args["timed"]) + "s_" + str(args["fps"]) + "fps" + "_raw" + ".mp4")  # FIX:CROSS PLATFORM
            videoName_path = os.path.join(dataOutput_path, videoName_common + "_raw" + ".mp4")
        else:
            # videoName_path = os.path.join(dataOutput_path, timeTag+"_"+args.get("condition")+ "_" + args['idlevel'] + "_" + categories + "_" + subjectID + "_" +controller +"_"+str(args["timed"])+"s_"+str(args["fps"])+"fps"+".mp4")# FIX:CROSS PLATFORM
            videoName_path = os.path.join(dataOutput_path, videoName_common + ".mp4")

    return videoName_path
    

####################################################
### save using panda dataframe  (slower)        ####
####################################################        
def save_dataframe(dataframe, x, args, timeTag, categories, gameTest, startTimeFormatted, note, dir_of_move, ballEscaped, obstacleHit, pathtype):
    if gameTest:
        controller = "Mouse"
    else:
        controller = "Tracking"
    #timeTag = time.strftime("%Y%m%d_%H%M%S")
    if args.get("video", False):
        #dataOutput_path = save_output_at("dataframeOutput_PP")
        subjectOnly= os.path.splitext(args["video"][0])[0]

        ## this part only for organizing postprocessing directory
        local_path = os.getcwd()
        dataOutput_path2 = os.path.join(str(local_path), "Output", subjectOnly, "dataframeOutput_PP" )
        if not os.path.exists(dataOutput_path2):
            os.makedirs(dataOutput_path2)


        filenameOnly= os.path.splitext(args["video"][1])[0]
        dataframe.to_csv(os.path.join(dataOutput_path2, filenameOnly + "_PP.csv"), sep=',', encoding='utf-8')

    else:
        dataOutput_path = save_output_at("dataframeOutput")
        # save as csv
        if  args["practice_boardtask"]: # add PR tp the name if it is practice trials
            fullfilename = os.path.join(dataOutput_path, timeTag + "_" + dir_of_move + "_" + args[
                'idlevel'] + "_" + categories + "_" +  args["subject"] + "_" + controller + "_success" + str(x) + "_PR" + ".csv")
        else:
            fullfilename = os.path.join(dataOutput_path, timeTag+"_"+dir_of_move+"_"+ args['idlevel'] + "_" +  categories + "_" +  args["subject"] + "_" + controller +   "_success"+str(x)+".csv")

        # print(dataframe.head(10))
        dataframe.to_csv(fullfilename, sep=',', encoding='utf-8')

        # write meta-data on top of the files. (hard attempt)
        with open(fullfilename, "w") as f:
            f.write('meta-data-length:'+ str(19) + '\n') #update as you add more meta-data.
            f.write('subjectID:' +  args["subject"] + '\n')
            f.write('timeTag:' + timeTag + '\n')
            f.write('startTime:'+startTimeFormatted+ '\n')
            f.write('mode:' + args["mode"] + '\n')
            f.write('code:' + categories + '\n')
            f.write('tasktype:' + args["tasktype"] + '\n')
            f.write('Direction:' +dir_of_move+ '\n')
            f.write('marker:' + args["marker"] + '\n')
            f.write('thread:' + str(args["thread"]) + '\n')
            f.write('display:' + str(args["display"]) + '\n')
            f.write('controller:' + controller + '\n')
            f.write('ID:' + args['idlevel'] + '\n')
            f.write('pathtype:' + pathtype + '\n')
            f.write('handedness:' + args['handedness'] + '\n')
            f.write('ballEscaped:' + str(ballEscaped) + '\n')
            f.write('obstacle:' + str(args["obstacles"]) + '\n')
            f.write('obstacleHit:' + str(obstacleHit) + '\n')
            f.write('Note:' + note + '\n')
            dataframe.to_csv(f, mode='a')

    return fullfilename
    

