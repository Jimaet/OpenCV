import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
confidence_threshold = 0.5
def captureAndSaveFace(frame, face_box, save_path='captured_face.jpg'):
    if face_box is not None:
        x1, y1, x2, y2 = face_box
        face_roi = frame[y1:y2, x1:x2]

        # Проверка, что лицо не является пустым
        if not face_roi.size == 0:
            face_resized = cv2.resize(face_roi, (1200, 1200))
            
            # Convert the face image to grayscale
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Save the grayscale image
            cv2.imwrite(save_path, face_gray)
        else:
            print("Лицо не обнаружено.")
    else:
        print("Лицо не обнаружено.")

def applyBackgroundBlur(frame, face_box, blur_factor=25):
    if face_box is not None:
        x1, y1, x2, y2 = face_box
        height, width, _ = frame.shape

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        mask = cv2.GaussianBlur(mask, (0, 0), blur_factor)
        mask = np.stack([mask] * 3, axis=-1)

        result_img = frame * mask + (1 - mask) * cv2.GaussianBlur(frame, (0, 0), blur_factor)
    else:
        result_img = frame

    return result_img

def findMainFace(face_boxes):
    if face_boxes:
        main_face = max(face_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        return main_face
    else:
        return None
def highlightMainFace(net, frame, conf_threshold=0.7):
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]

    blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), (104, 117, 123), True, False)
    net.setInput(blob)
    detections = net.forward()

    main_face = None  # Initialize variable to store main face

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)

            face_area = (x2 - x1) * (y2 - y1)

            # Check if the current face is the largest face detected
            if main_face is None or face_area > main_face[2]:
                main_face = (x1, y1, face_area, x2, y2)

    # Draw rectangle only around the main face
    if main_face is not None:
        x1, y1, _, x2, y2 = main_face
        cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)

    return frame_opencv_dnn, [main_face] if main_face is not None else []
# Add this line to your code to initialize the face detector
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
face_net = cv2.dnn.readNet(faceModel, faceProto)

model = load_model('main_model_for_opencv.h5')

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cv2.waitKey(1) < 0:
    has_frame, frame = video.read()

    if not has_frame:
        cv2.waitKey()
        break

    result_img, face_boxes = highlightMainFace(face_net, frame)
    main_face = findMainFace(face_boxes)

    if main_face is not None:
        if len(main_face) == 5:
            x1, y1, _, x2, y2 = main_face
            face_roi = frame[y1:y2, x1:x2]

            if not face_roi.size == 0:
                face_resized = cv2.resize(face_roi, (224, 224))

                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

                face_rgb = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2RGB)

                face_normalized = face_rgb.astype('float32') / 255.0
                face_normalized = np.expand_dims(face_normalized, axis=0)

                predictions = model.predict(face_normalized)
                predicted_class = np.argmax(predictions)
                print(f"Predicted class: {predicted_class}")
                max_confidence = np.max(predictions)

                if max_confidence <= confidence_threshold:
                    print("Неизвестный класс")
                    print(f"Уверенность: {max_confidence}")
                else:
                    print("Есть в базе")
            print(predictions)

    cv2.imshow("Face detection", result_img)