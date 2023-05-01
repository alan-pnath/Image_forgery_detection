from django.http import HttpResponse
from django.shortcuts import render
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance
import matplotlib.pyplot as plt

class_names = ['Forged', 'Authentic']
model=load_model("train/trained_model1.h5")

def home(request):
    if request.method == 'POST':
        uploaded_image = request.FILES['input_image']
        test_image = prepare_image(uploaded_image)
        test_image = test_image.reshape(-1, 128, 128, 3)
        y_pred = model.predict(test_image)
        y_pred_class = round(y_pred[0][0]) 


        print(f'Prediction: {class_names[y_pred_class]}')
        prediction=class_names[y_pred_class]
        if y_pred<=0.5:
            print(f'Confidence:  {(1-(y_pred[0][0])) * 100:0.2f}%')
            confidence=f'{(1-(y_pred[0][0])) * 100:0.2f}'
        else:
            print(f'Confidence: {(y_pred[0][0]) * 100:0.2f}%')
            confidence=f'{(y_pred[0][0]) * 100:0.2f}'
        print('--------------------------------------------------------------------------------------------------------------')
        return render(request, 'result.html',{'pred':prediction,'con':confidence})

    return render(request, 'base.html')


                                                                                                                                                                            
def prepare_image(image_path):
    image_size = (128, 128)
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0         #normalizing the array values obtained from input image

def convert_to_ela_image(path,quality):

    original_image = Image.open(path).convert('RGB')

    #resaving input image at the desired quality
    resaved_file_name = 'resaved_image.jpg'     #predefined filename for resaved image
    original_image.save('static/predicted/resaved_image.jpg','JPEG',quality=quality)
    resaved_image = Image.open('static/predicted/resaved_image.jpg')

    #pixel difference between original and resaved image
    ela_image = ImageChops.difference(original_image,resaved_image)
    
    #scaling factors are calculated from pixel extremas
    extrema = ela_image.getextrema()
    max_difference = max([pix[1] for pix in extrema])
    if max_difference ==0:
        max_difference = 1
    scale = 350.0 / max_difference
    
    #enhancing elaimage to brighten the pixels
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    ela_image.save("static/predicted/ela_image.png")
    return ela_image