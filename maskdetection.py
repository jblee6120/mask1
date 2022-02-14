from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

#마스크 인식을 위한 네트워크 불러오기
facenet = cv2.dnn.readNet('models/deploy.prototxt.txt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
model = load_model('models/mask_detector.model')

cap = cv2.VideoCapture(0)
ret, img = cap.read()

#fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#out = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (img.shape[1], img.shape[0]))

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    h, w = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.)) #네트워크 입력 blob 만들기
    facenet.setInput(blob) #readNet 객체에 네트워크 입력하기 = 인풋 데이터 입력
    dets = facenet.forward() #네트워크 순방향 추론 진행하기 = 결과추론

    result_img = img.copy() 

    for i in range(dets.shape[2]): #얼굴 검출해서 검사하기
        confidence = dets[0, 0, i, 2] #검출시 확률
        if confidence < 0.5:
            continue

        x1 = int(dets[0, 0, i, 3] * w) #검출된 얼굴 사진의 테두리 설정
        y1 = int(dets[0, 0, i, 4] * h)
        x2 = int(dets[0, 0, i, 5] * w)
        y2 = int(dets[0, 0, i, 6] * h)
        
        face = img[y1:y2, x1:x2] #얼굴 사진만 잘라내서 저장

        face_input = cv2.resize(face, dsize=(224, 224)) #이미지 크기를 바꿔준다
        face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB) #순서를 RGB로 바꾼다
        face_input = preprocess_input(face_input) #전처리를 해준다
        face_input = np.expand_dims(face_input, axis=0) #input이 (1, 244, 244, 3)되어야 함. 그래서 face_input 0번 axis에 차원을 하나 추가해준다.
        
        mask, nomask = model.predict(face_input).squeeze() #mask를 썼는지 안 썼는지 예측

        if mask > nomask: #mask를 썼다면 초록색 글씨로 마스크 착용확률을 표시
            color = (0, 255, 0)
            label = 'Mask %d%%' % (mask * 100)
        else: #mask를 착용하지 않았다면 빨간색 글씨로 미착용 확률을 표시
            color = (0, 0, 255)
            label = 'No Mask %d%%' % (nomask * 100)

        cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
        cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2, lineType=cv2.LINE_AA)

    #out.write(result_img)
    cv2.imshow('result', result_img)
    if cv2.waitKey(1) == ord('q'):
        break
print(x1, y1, x2, y2)
print(img.shape)
#out.release()
cap.release()