
# Pupil & Iris Detection

## Description
This project focuses on Pupil and Iris detection using Canny Edge Detection in videos.\
It makes use of OpenCV-python with its built-in functionalities.

## Data
The program Detection.py makes use Eye-Video.mov file as an input which is available here https://drive.google.com/file/d/1_Je8h0i6fMAwfdNQo9q9Mvi8DT946SsA/view?usp=sharing or the video can be downloaded from the directory.\
Change the video file path in the code according to particular system path that you are working on.

## Detection Process
The detection of the Pupil and Iris involves 4 major steps - 
1. Blurring the frame
2. Converting the frame to grayscale
3. Applying thresholding to distinguish between Iris and the Pupil
4. Applying canny edge detector on the given frame.
