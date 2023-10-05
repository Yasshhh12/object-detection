from tkinter import Frame
from types import FrameType
import cv2
import numpy as np
from matplotlib.patches import Polygon

#opencv DNN

net = cv2.dnn.readNet(r"C:\Users\yasht\Downloads\dnn_model-220107-114215\dnn_model\yolov4-tiny.weights",r"C:\Users\yasht\Downloads\dnn_model-220107-114215\dnn_model\yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416,416),scale=1/255)

#load class lists
classes = []
with open("D:\Object detection\dnn_model\classes.txt","r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("Object List")
print(classes)

#intialise camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

button_person = False

#full hd 1920*1080

def click_button(event, x, y, flags, params):

    
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        Polygon = np.array([[(20,20) ,(220,20), (220,70), (20,70)]])
        
        is_inside = cv2.pointPolygonTest(Polygon, (x,y), False)
        if is_inside > 0:
            print("We are clicking this button", x, y)

            if button_person is False:
                button_person = True
            else:
                button_person = False

            print()



#create window
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)
while True:
    #get frames
    ret, Frame = cap.read()

    #object detection
    (class_ids, scores, bboxes)= model.detect(Frame)
    for class_id, score, bbox in zip(class_ids, scores,bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]

        cv2.putText(Frame, class_name, (x,y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200,0,50), 2)
        print(x, y, w, h)
        cv2.rectangle(Frame, (x,y), (x+w,y+h), (200, 0, 50), 3)
#    print("class ids",class_ids)
#    print("bboxes",bboxes)
    #create buttons
    cv2.rectangle(Frame, (20,20), (220,70), (0, 0, 200), -1)
    Polygon = np.array([[(20,20) ,(220,20), (220,70), (20,70)]])
    cv2.fillPoly(Frame, Polygon, (0, 0, 200))
    cv2.putText(Frame, "Person", (30,60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    cv2.imshow("Frame",Frame)
    cv2.waitKey(1)