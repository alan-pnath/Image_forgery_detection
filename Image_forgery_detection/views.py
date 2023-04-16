from django.http import HttpResponse
from django.shortcuts import render
import cv2
import numpy as np
from keras.models import load_model


def home(request):
    if request.method == 'POST':
        uploaded_image = request.FILES['input_image']


        # Set the BDCT block size
        block_size = 8

        # Load the image and convert it to grayscale
        img = cv2.imread("uploaded_image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Convert the image to float32 data type
        img = img.astype(np.float32)

        # Compute the BDCT coefficients for each block
        bdct = np.zeros(img.shape)
        for i in range(0, img.shape[0], block_size):
            for j in range(0, img.shape[1], block_size):
                bdct[i:i+block_size, j:j+block_size] = cv2.dct(
                    img[i:i+block_size, j:j+block_size])

        # Compute the threshold using the mean and standard deviation of the BDCT coefficients
        threshold = np.mean(bdct) + np.std(bdct)

        # Apply the threshold to the BDCT coefficients to obtain a binary mask
        mask = (bdct > threshold).astype(np.uint8)

        # Resize the mask to match the input size of the CNN
        mask = cv2.resize(mask, (224, 224))

        # Stack the mask together as input to the CNN
        X = np.expand_dims(mask, axis=-1)
        X = np.expand_dims(X, axis=0)

        # Load the trained model
        model = load_model("image-forgery-detection-main/trained_model1.h5")

        # Predict the authenticity of the image
        prediction = model.predict(X)

        if prediction[0][0] > 0.5:
            print("The image is authentic.")
        else:
            print("The image is forged.")



    return render(request, 'base.html')
