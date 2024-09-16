Demo for Live Lead Migration Detection  
====================================
---

``` 

Lead Migration Detection Algorithm Developer:  Hyunjoo Park
Code Author: Won Joon Sohn (wonjsohn@gmail.com)
Current appointment: 
Related Publication: 
```

---
``` Modification log: ```

--- 
Application: 
* Live Demo for Lead migration Detection 
```


Requirements
------------
Build in python3.10
```bash
pip install pygame numpy ... (many more)
```
* For windows system, download the python 3.10 installer (web-based, or any other) and add python36 to PATH.  


How to operate (in command line for Live Lead Migration, ignore the instructions below )
------
1. Run as: `>python main.py` + options.  (e.g. `>python main.py -mod "play")




How to operate (in command line)
-------
1. Open the MagicTable src folder.
2. Run as: `>python main.py` + options.
3. Option are play, pp, pygame, realtime_plot in broad category.
    1. play mode: normal mode 
    2. pp mode: post-processing mode (work with pre-recorded videos)
    3. pygame mode: game mode. You have keyboard  
    4. Realtime plot: for realtime plotting the energy margin. 
```bash
BoardTask   e.g. > main.py -mod "dual" -tt "p2p" -sid 'subjectID' -hn "r" -idx "ID1" -obs 0 -t 30
Pygame      e.g. > main.py -mod "pygame" -sid "computer0" -pt 1
```
4. SubjectID. Mapping of subjects name - ID in encryped file (subject_mapping.enc).
    * computer0 : any testing trials 
    * NTxxx     : Neurotypicals
    * ASDxxx    : ASD 
5. Position checker - press 'S' to retake the snapshot. Otherwise esc to skip the snapshot. 
6. Snapshot taker - with cup removed from the scene, press space bar to detect the shapes until satisfied, press esc to move on.  
7.  Press esc to break out of current trial. The trial will still be saved.
8.  Press 'D' or 'd' to delete the current trial. It will break out of the trial and not save. 
9.  Press 'C' or 'c' to when the goal is reached in the fig8 task.

Snapshot of PyCharm Project run/debug configuration (as of 2018.11)
![Libraries](resources/PyCharm_runconfig.png?raw=true)


How to play in GUI : a preferred way
-------
![Libraries](resources/GUI_play.png?raw=true)
1. In ../graphcial_panel/ sub-directory, run MAGIcTableGUI_Runner.py.  For windows, this can be done by clicking the batch file OneClick_MAGIC_TABLE.bat in the same directory.
2. GUI window with argument options will be presented. Click "Run" button after selecting options. Instructions will pop-up in the camera-alignment and snapshot stage.        
3. Selected options are automatically saved even if the current GUI window is closed.  


### Other options (check arguments.py)
* **subject** [-sid]: subject ID. 
* **note** [-nt]: leave notes for this specific trial
* **handedness**[-hn]: Righthander:r (default), lefthander:l 
* **timed** [-t]: Set the operating time
* **codec** [-cod]: Specify the codec of the output video file
* **thread** [-th]: Whether or not to use thread in reading frames
* **rawvideo** [-rv]: Whether you want to record raw video. (This is the ultimate back-up data) in case real-time detection fails.
 
#### Tracking
* **marker** [-m]: Specify the tracking algorithm. (e.g. el_object, cl_object)



#### Display
* **display** [-d]: Whether or not the webcam frames should be displayed
* **virtual** [-vir]: Add virtual display (For augmented reality)
* **clock**[-clk]: Display clock and frame number
* **targetsound** [-ts]: sound on at target
* **targetvisual** [-tv]: highlight if cursor on the target
* **trace** [-tr]:Whether or not  traces should be displayed
* **linetrace** [-ltr]: Whether or not line traces should be displayed

## Important files
* **arguments.py**: Sepcify input parameters to the main.py program
* **main.py**: The central loop for the camera tracking system.
* **save.py**: All the saving related functions.
* **snapshots.py**: For taking a snapshot of a board from webcam.
* **shape_detection.py**:  detect targets and obstacles when you first register the board.
* **colorRangeDetector.py**: Used to tune the color filter indices. If the lighting condition changes, it may be necessary to tune the indices.
* **check_camera_position.py**: the first file to be run (in main.py) to check the camera position.
* **graphical_panel/MAGIcTableGUI_Runner.py**:  starts the GUI. One Click Batch file runs this file.  
* **graphical_panel/OneClick_MAGIC_TABLE.bat**: (Windows-only for batch file) One click activation of GUI. 
* **games**: game related file packaged per game.

## Tips
* To increase fps, minimize display. If audio-feedback only, it is the fastest realtime performance.
* Virtual display mode will enable many gaming options.   
* You only have to set only the "mode" without many suboptions.
* **Performance**: as of 2018.11 by Won Joon, 160 FPS in Windows Desktop (i7-7700, RAM 512MB, SSD), 100 FPS in Mobile Surface pro 4 for single object tracking.  For dual-object tracking, 105 FPS in Windows Desktop. For some reason, performance in MAC was slower.     
* For each subject, set the subject ID and handedness. (e.g. main.py -mod "play" -tt "p2p" -sid "NT0" -hn "r" -t 30)
* It helps to speed up the process by using "run configuration" in the pycharm which makes running each trial a one click. 

## Magic table file structure

MAGIC_TABLE_Root
    |_ graphical_panel
    |_ resouces
        |_ images
        |_ sounds
    |_ Output: (time, x, y, xb, yb) 
        |_videoOutput
        |_snapshots
        |_pickles
        |_dataframeOutput



General output filename convention: [timestamp_mode_subjectID_success.csv]
e.g. 20180910_134223_NT0_success.csv


## Additional files

* videoOutput: recorded video from the camera during trials. Filename: [timestamp_mode_timeDuration_fps.mp4]
    * TODO: change mode name  
* snapshots: snapshots taken. Filename: [timestamp.jpg]
* pickles: timestamp_circles.dump (centers for two circles) / timestamp_rectangles.dump (if any rec is drawn) 
    * TODO: combine the two in one file. 
    
## Output folders 

* **videoOutput**: recorded video from the camera during trials. Filename: [timestamp_mode_timeDuration_fps.mp4]
* **snapshots**: snapshots taken. Filename: [timestamp.jpg]
* **pickles**: timestamp_circles.dump (centers for two circles) / timestamp_rectangles.dump (if any rec is drawn) 


Output video format (codec options)
---------------------
```
# -*- coding: utf-8 -*-
– / avi / 112512 kB / I420 / WMP, VLC, Films&TV, MovieMaker
MJPG / avi / 14115 kB / MJPG / VLC
MJPG / mp4 / 5111 kB / 6C / VLC
CVID / avi / 7459 kB / cvid / WMP, VLC, MovieMaker
MSVC / avi / 83082 kB / CRAM / WMP, VLC
X264 / avi / 187 kB / H264 / WMP, VLC, Films&TV, MovieMaker  => doesn't work in Windows...
XVID / avi / 601 kB / XVID / WMP, VLC, MovieMaker
XVID / mp4 / 587 kB / 20 / WMP, VLC, Films&TV, MovieMaker
PIM1 / mp4 / similar to XVID /     MPEG-1 Codec
```

** There could be "OpenCV FFMPEG" related warning message with a codec 'XVID'. This is about codec and extension matching.  This warning can be ignored if you care less about output video file size not being optimally small.   

