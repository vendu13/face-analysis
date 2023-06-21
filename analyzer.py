import cv2
from deepface import DeepFace
import face_recognition as fr
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np

# img1 = cv2.imread('Images/elon musk.jpg')
#
# result = DeepFace.analyze(img1, actions=['age', 'gender', 'race', 'emotion'])
# print(type(result))

class face_cod:
    def extract_face_cod(self, img):
        face_cod = fr.face_locations(img)
        return face_cod[0]

class module_1:

    def analysis(self, image_path):
        try:
            img = cv2.imread(image_path)
            raw = DeepFace.analyze(img, actions=['age', 'gender', 'emotion'])
            if raw['age'] <= 15:
                age = '<= 15'
            elif raw['age'] <= 20:
                age = '<= 20'
            elif raw['age'] <= 30:
                age = '<= 30'
            elif raw['age'] <= 40:
                age = '<= 40'
            elif raw['age'] <= 50:
                age = '<= 50'
            elif raw['age'] <= 60:
                age = '<= 60'
            else:
                age = '>70'

            result = 'Age: ' + age + '\nGender: ' + raw['gender'] + '\nDominant Emotion: ' + raw['dominant_emotion']
        except:
            result = 'Age: ' + 'NA' + '\nGender: ' + 'NA' + '\nDominant Emotion: ' + 'NA'
        return result

class module_2:
    prototxtPath = r"face_detector\deploy.prototxt"
    weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    specsNet = load_model("specs_detector.model")
    maskNet = load_model("mask_detector.model")

    def detect_and_predict_specs_mask(self, image_path):
        frame = cv2.imread(image_path)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                     (104.0, 177.0, 123.0))

        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        print(detections.shape)

        faces = []
        locs = []
        preds_specs = []
        preds_mask = []

        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
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
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds_specs = self.specsNet.predict(faces, batch_size=32)
            preds_mask = self.maskNet.predict(faces, batch_size=32)

        # return a 2-tuple of the face locations and their corresponding
        # locations
        mask = 0
        no_mask = 0
        specs = 0
        no_specs = 0
        for p in preds_mask:
            (mask, no_mask) = p
        for p in preds_specs:
            (no_specs, specs) = p
        mask_pred_result = "Wearing Mask" if mask > no_mask else "Not Wearing Mask"
        specs_pred_result = "Wearing Specs" if specs > no_specs else "Not Wearing Specs"
        result = '\nMask: '+ mask_pred_result + '\nSpecs: '+specs_pred_result
        return result

class module_3:
    prototxtPath = r"face_detector\deploy.prototxt"
    weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    beardNet = load_model("beard_detector.model")
    maskNet = load_model("hair_detector.model")

    def detect_and_predict_hair_beard(self, image_path):
        frame = cv2.imread(image_path)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                     (104.0, 177.0, 123.0))

        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        print(detections.shape)

        faces = []
        locs = []
        preds_beard = []
        preds_hair = []

        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
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
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds_beard = self.beardNet.predict(faces, batch_size=32)
            preds_hair = self.maskNet.predict(faces, batch_size=32)

        # return a 2-tuple of the face locations and their corresponding
        # locations
        hair = 0
        no_hair = 0
        beard = 0
        no_beard = 0
        for p in preds_hair:
            (mask, no_mask) = p
        for p in preds_beard:
            (beard, no_beard) = p
        hair_pred_result = "Hair Person" if mask > no_mask else "Bald Person"
        beard_pred_result = "Having Beard" if beard > no_beard else "No Beard"
        result = '\nHair: ' + hair_pred_result + '\nBeard: ' + beard_pred_result
        return result

#
# obj = module_1()
# print(obj.analysis("Images\wanda.jpg"))

# cv2.imshow('Lable', img1[26:51, 27:51])
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()