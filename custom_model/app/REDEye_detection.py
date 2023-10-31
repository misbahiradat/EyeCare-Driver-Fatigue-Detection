# Author @ Misbah Iradat
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x + w, y + int(h/2)), (255, 0, 0), 5)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)
            for (ex, ey, ew, eh) in eyes:
                eyes_frame = roi_color[ey:ey+eh, ex:ex+ew]
                name = 'eye_frame.jpg'
                image = cv2.imwrite(name, eyes_frame)

            new_size = (128, 128)
            n_image = Image.open('eye_frame.jpg')
            final_img = n_image.resize(new_size)
            #final_img = cv2.resize(image, (128,128))
            final_img.save('resized.jpg')
            final_img = np.expand_dims(final_img, axis=0)            
            final_img = final_img / 255 

            new_model = load_model('../Model_Outputs/2022_08_25/test1/model/model.h5', compile=False)
            Predictions = new_model.predict(final_img)
            y_classes = Predictions.argmax(axis=-1)
            print(y_classes)

        if y_classes == 0:
            status = 'Normal_Eye'
        else:
            status = 'Red_Eye'

        font=cv2.FONT_HERSHEY_SIMPLEX

        #inserting text on video
        cv2.putText(frame,
                    status,
                    (50,50),
                    font,3,
                    (0,0,255),
                    2,
                    cv2.LINE_4)


    cv2.imshow('Red_eye_detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()