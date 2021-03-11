# import the necessary packages
import sys
from base_camera import BaseCamera
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
from mask_detection import *
import time
import cv2
import os

class Camera(BaseCamera):
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # basedir = os.path.abspath(os.path.dirname(__file__))
    confidence_arg = 0.5
    
    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = "deploy.prototxt"
    weightsPath = "res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    
    maskNet = load_model("mask_detector.model")

    def detect_and_predict_mask(frame, faceNet, maskNet):
            Camera.confidence_arg = 0.5
            # grab the dimensions of the frame and then construct a blob
            # from it
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                (104.0, 177.0, 123.0))

            # pass the blob through the network and obtain the face detections
            faceNet.setInput(blob)
            detections = faceNet.forward()

            # initialize our list of faces, their corresponding locations,
            # and the list of predictions from our face mask network
            faces = []
            locs = []
            preds = []

            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the detection
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the confidence is
                # greater than the minimum confidence
                if confidence > Camera.confidence_arg:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # ensure the bounding boxes fall within the dimensions of
                    # the frame
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                    # extract the face ROI, convert it from BGR to RGB channel
                    # ordering, resize it to 224x224, and preprocess it
                    face = frame[startY:endY, startX:endX]
                    preds.append((detect_face_mask(face)))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)

                    # add the face and bounding boxes to their respective
                    # lists
                    faces.append(face)
                    if abs(endY-startY) > 40:
                        locs.append((startX, startY, endX, endY))

            # only make a predictions if at least one face was detected
            # if len(faces) > 0:
            #     # for faster inference we'll make batch predictions on *all*
            #     # faces at the same time rather than one-by-one predictions
            #     # in the above `for` loop
            #     faces = np.array(faces, dtype="float32")
            #     preds = maskNet.predict(faces, batch_size=32)


            # return a 2-tuple of the face locations and their corresponding
            # locations
            return (locs, preds)


    @staticmethod
    def frames():    

        # loop over the frames from the video stream
        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            frame = Camera.vs.read()
            frame = imutils.resize(frame, width=800)

            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            (locs, preds) = Camera.detect_and_predict_mask(frame, Camera.faceNet, Camera.maskNet)

            # loop over the detected face locations and their corresponding
            # locations
            temp_list = []
            for i, (box, pred) in enumerate(zip(locs, preds)):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                temp_list.append([i, abs(endY - startY)])
                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            result = None
            if temp_list:
                top_index_to_pick = sorted(temp_list, key=lambda x: x[1], reverse=True)[0][0]
                mask, withoutMask = preds[top_index_to_pick]
                result = 'mask' if mask > withoutMask else 'nomask'
            # convert image to jpg format
            ret, jpeg = cv2.imencode('.jpg', frame)

            yield jpeg, result