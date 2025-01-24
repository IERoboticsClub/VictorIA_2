import os
import tensorflow
import cv2
import numpy as np
#import tensorflow as tf
#from tensorflow.python.keras.layers import load_model
#from keras.src.utils.module_utils import tensorflow
#from tensorflow.python.keras.models import load_model
#from teachable_machine import TeachableMachine
#import tensorflow.keras
# remember tensorflow 2.12 needed



class CircleRecognition:
    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        # suppress TensorFlow debugging logs (annoying)
        tensorflow.keras.utils.disable_interactive_logging()

        #self.model = load_model(os.path.join(base_path, "../Models/teachable/keras_model.h5"), compile=False)
        self.model = tensorflow.keras.models.load_model(os.path.join(base_path, '../Models/teachable/keras_model.h5'))
        #self.model = TeachableMachine(model_path=os.path.join(base_path, "../Models/teachable/keras_model.h5"),
        #                 labels_file_path=os.path.join(base_path, '../Models/teachable/labels.txt'))
        self.class_names = open(os.path.join(base_path, "../Models/teachable/labels.txt"), "r").readlines()

    def predict(self, image):
        # resize the raw image into (224-height,224-width) pixels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        # display image for debugging
        #cv2.imshow("Image", image)

        # make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # normalize the image array
        image = (image / 127.5) - 1


        # predicts the model
        prediction = self.model.predict(image, verbose=0)
        index = np.argmax(prediction)
        class_name = self.class_names[index]
        confidence_score = prediction[0][index]

        if "detection_boxes" in prediction:
            bounding_boxes = prediction["detection_boxes"][0]  # Extract bounding boxes for the first detected object
            print(f"Bounding Box: {bounding_boxes}")  # Debug print for bounding box
        else:
            print("no boxes")
            bounding_boxes = None

        # Print prediction and confidence score
        #print("Class:", class_name[2:], end="")
        #print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

        """
        result = self.model.classify_image("rectified_image.jpg")

        print("class_index", result["class_index"])

        print("class_name:::", result["class_name"])

        print("class_confidence:", result["class_confidence"])

        print("predictions:", result["predictions"])

        return result["class_name"], result["class_confidence"]
        """

        return class_name, confidence_score


cr = CircleRecognition()
image = cv2.imread(f"../cells/Empty Cells/cell_0_0.jpg")
cr.predict(image)