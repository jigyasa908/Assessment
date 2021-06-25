# -*- coding:utf-8 -*-
"""
Author - Jigyasa Singh
Take Out 2 - Pupil and Iris Detection
This program focuses on Pupil and Iris detection in a video file using OpenCV-python built-in functions
"""
import cv2
video_input = cv2.VideoCapture('Eye-Video.mov')
while video_input.isOpened():
    ret, frame = video_input.read()
    blur_frame = cv2.GaussianBlur(frame, (7, 7), 0)
    gray_frame = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2GRAY)
    # gray_frame = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2GRAY)
    ret, pupil_frame = cv2.threshold(gray_frame, 35, 255, cv2.THRESH_BINARY_INV)
    ret, iris_frame = cv2.threshold(gray_frame, 100, 255, cv2.THRESH_BINARY_INV)
    # pupil_frame = cv2.bitwise_and(frame, frame, mask=pupil_frame)
    # iris_frame = cv2.bitwise_and(frame, frame, mask=iris_frame)
    pupil_edge = cv2.Canny(pupil_frame, 25, 75)
    iris_edge = cv2.Canny(iris_frame, 25, 50)
    cv2.imshow('Pupil Detection', pupil_edge)
    cv2.imshow('Iris Detection', iris_edge)
    if cv2.waitKey(2) == ord('q'):
        break
