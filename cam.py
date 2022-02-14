import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import os
import serial
import time

port = 'COM6'
baudrate = 9600
frame_no = 0
ser = serial.Serial(port, baudrate)


def send_result(wear, nowear):
    if ser.readable():
        
        if wear > nowear:
            pre_result = '1'
            pre_result = pre_result.encode('utf-8')
            ser.write(pre_result)
        
        elif wear < nowear:
            pre_result = '0'
            pre_result = pre_result.encode('utf-8')
            ser.write(pre_result)
           


cap = cv2.VideoCapture(0)

net = cv2.dnn.readNet('models/deploy.prototxt.txt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
model = load_model('models/mask_detector.model')


while cap.isOpened() and ser.readable():
    ret, frame = cap.read()
    #frame_no = frame_no + 1
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
    net.setInput(blob)
    
    pre = net.forward()
    result = frame.copy()

    for i in range(pre.shape[2]):
        confidence = pre[0, 0, i, 2]

        if confidence < 0.6:
            continue
            
        x0 = int(pre[0, 0, i, 3] * width)
        y0 = int(pre[0, 0, i, 4] * height)
        x1 = int(pre[0, 0, i, 5] * width)
        y1 = int(pre[0, 0, i, 6] * height)

        face = frame[y0:y1, x0:x1]

        input_frame = cv2.resize(face, (224, 224))
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        input_frame = preprocess_input(input_frame)
        input_frame = np.expand_dims(input_frame, axis=0)
            
        mask, nomask = model.predict(input_frame).squeeze()

        if mask > nomask : #and frame_no % 100 == 0:
            color = (0, 255, 0)
            label = 'mask checked %d%%' % (mask * 100)
            #send_result(mask, nomask)
            pre_result = '1'
            pre_result = pre_result.encode('utf-8')
            ser.write(pre_result)
                
            #print(pre_result)

        elif mask < nomask : #and frame_no % 100 == 0:
            color = (0, 0, 255)
            label = 'please wear a mask %d%%' % (nomask * 100)
            #send_result(mask, nomask)
            pre_result = '0'
            pre_result = pre_result.encode('utf-8')
            ser.write(pre_result)
                
            #print(pre_result)
        
        cv2.rectangle(result, pt1=(x0, y0), pt2=(x1, y1), thickness=2, color=color, lineType=cv2.LINE_AA)
        cv2.putText(result, text=label, org=(x0, y0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=color, lineType=cv2.LINE_AA)
        cv2.imshow('result', result)

    if cv2.waitKey(10) == 27: #opencv 캡쳐 주기가 아두이노 delay와 맞아야 함. 맞지 않으면 아두이노 수신 버퍼에 데이터가 쌓임. 현재 상태에 대한 리액션이 나오지 않음.
        break
    
    """elif cv2.waitKey(1) == 32 and mask > nomask:
        #time.sleep(1)
        pre_result = '1'
        print(pre_result)
        #pre_result = pre_result.encode('utf-8')
        #ser.write(pre_result)

    elif cv2.waitKey(1) == 32 and mask < nomask:
        #time.sleep(1)
        pre_result = '0'
        print(pre_result)
        #pre_result = pre_result.encode('utf-8')
        #ser.write(pre_result)
    """


cap.release()
cv2.destroyAllWindows()
