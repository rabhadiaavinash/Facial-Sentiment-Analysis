import cv2
import numpy as np
import cvlib as cv
from matplotlib import pyplot as plt
import tensorflow as tf

min_loss_model = tf.keras.models.load_model(r"D:\Facial Emotions Recognition\Face_detections_emotions EfficientNet\min_loss.h5")

cls_dict = {0:'angry',1:'happy',2:'neutral',3:'sad',4:'surprise'}


target_size = (96, 96)
interpolation = 'bilinear'
color_mode = 'grayscale'
batch_size = 32

list_of_frames = []
list_of_classes = []

def show_detection(image, faces,list_of_classes):
    print("inside show_detection")
    counter = 0
    print("len of faces:",len(faces))
    print("list_of_classes:",list_of_classes)

    if len(faces) == len(list_of_classes):
        print("Emotions")
        for (startX, startY, endX, endY) in faces:
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 3)
            cls_name= list_of_classes[counter]
            cv2.putText(image,cls_name,(startX-10,startY-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            counter += 1
    return image

def detect_faces(list_of_frames):

    faces_list = []
    list_of_roi = []
    list_of_classes = []
    for fr in list_of_frames:
        faces, confidences = cv.detect_face(fr)
        faces_list = []
        for (startX, startY, endX, endY) in faces:
            #list_of_roi = []
            list_of_classes = []
            w = endX - startX
            h = endY - startY
            Roi_cropped = fr[startY:startY + h, startX:startX + w]
            try:

                if Roi_cropped is None:
                    print("No ROI")
                else:
                    Roi_cropped1 = cv2.cvtColor(Roi_cropped, cv2.COLOR_BGR2GRAY)
                    Roi_cropped2 = cv2.resize(Roi_cropped1, (96, 96))
                    list_of_roi.append(tf.keras.preprocessing.image.img_to_array(Roi_cropped2))
                    faces_list.append(faces)
            except:
                print("ERROR")
                list_of_roi = []
                faces_list = []
                list_of_roi = []
                list_of_classes = []

        print('len of list_of_roi:',len(list_of_roi))
        if len(list_of_roi) != 0:
            list_of_roi_arr = np.array(list_of_roi)
            y_prediction = min_loss_model.predict(list_of_roi_arr)
            print("y_prediction:",y_prediction)
            #print("Type Y_pred:",type(y_prediction))
            y_prediction1 = list(y_prediction)
            for y_pred in y_prediction1:
                index = np.argmax(y_pred)
                class_name = cls_dict[index]
                #print(class_name)
                list_of_classes.append(class_name)

            img = show_detection(fr, faces_list[0], list_of_classes)
            faces_list = []
            list_of_roi = []
            list_of_classes = []
            cv2.imshow('Emotions', img)
            result.write(img)

        else:
            faces_list = []
            list_of_roi = []
            list_of_classes = []




# Create a VideoCapture object and read from input file
# cap = cv2.VideoCapture(r"D:\Facial Emotions Recognition\Face_detections_emotions EfficientNet\sentiment1.mp4")
cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print("fps",fps)
result = cv2.VideoWriter('output5.avi',
                             cv2.VideoWriter_fourcc(*'XVID'),
                             5, (2500,1400))

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video file")

# Read until video is completed
while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame,(2500,1400))
    if ret == True:
        if len(list_of_frames) != 2:
            list_of_frames.append(frame)
        else:

            detect_faces(list_of_frames)
            list_of_frames = []


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()
result.release()

# Closes all the frames
cv2.destroyAllWindows()
