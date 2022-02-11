import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class alzheimer:
    def __init__(self,filename):
        self.filename =filename

    def predictionalzheimer(self):
        # load model
        model = load_model('model.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        if result[0][0] == 1:
            prediction = 'ad'
            print(prediction)
        elif result[0][1] == 1:
            prediction = 'cn'
            print(prediction)
        else:
            prediction = 'mci'
            print(prediction)
        return prediction