import numpy as np
import cv2


import matplotlib.pyplot as plt

config_file= 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

frozen_model='frozen_inference_graph.pb'

model =cv2.dnn_DetectionModel(frozen_model,config_file)

classLabels = []
file_name = 'labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')


syrup_img = r'C:/Users/kvjai/spyder projects/medicine hidden cost detector/data/cough.jpeg'
tablet = r"C:/Users/kvjai/spyder projects/medicine hidden cost detector/data/tablet.jpeg"
injection = r"C:/Users/kvjai/spyder projects/medicine hidden cost detector/data/injection.jpeg"
capsule = r"C:/Users/kvjai/spyder projects/medicine hidden cost detector/data/capsule.jpeg"
leaf = r"C:/Users/kvjai/spyder projects/medicine hidden cost detector/data/leaf.jpeg"

img=cv2.imread(leaf)
plt.imshow(img)
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

#Threshold setup
thres = 0.3 # Threshold to detect object
nms_threshold = 0.2

#standard configuration setting up
classFile = r"C:/Users/kvjai/spyder projects/medicine hidden cost detector/data/coco.names"


#cap = cv2.VideoCapture(0) #use camera 0 

classNames = []
with open(classFile,"rt") as f:
        classNames = f.read().rstrip("\n").split("\n")


    #success,img = cap.read()
    

classIds, confs, bbox = model.detect(img,confThreshold=thres)
bbox = list(bbox)
confs = list(np.array(confs).reshape(1,-1)[0])
confs = list(map(float,confs))
    #print(type(confs[0]))
    #print(confs)

indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
    #print(indices)

for i in indices:
    print('i is: ',i)
        #i = i[0]
    box = bbox[i]
    x,y,w,h = box[0],box[1],box[2],box[3]
    cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
    cv2.putText(img,classNames[classIds[i]-1].upper(),(box[0]+10,box[1]+30),
    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

cv2.imshow("Output",img)
cv2.waitKey(1)
cv2.imshow("Output",img)
cv2.waitKey(1)

