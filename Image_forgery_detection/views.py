from django.http import HttpResponse
from django.shortcuts import render
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance
import matplotlib.pyplot as plt
import io
import imghdr
import requests
import json
from sightengine.client import SightengineClient


class_names = ['Forged', 'Authentic']
model=load_model("train/trained_model1.h5")

def home(request):
    data={}
    if request.method == 'POST':
        uploaded_image = request.FILES['input_image']
        file_type = imghdr.what(uploaded_image)
        if file_type is not None:
            img = Image.open(io.BytesIO(uploaded_image.read()))
        
            if img.format in ['JPEG', 'PNG']:
                test_image = prepare_image(uploaded_image)
                # params = {
                # 'models': 'properties,scam',
                # api_user: '1641196078',
                # api_secret: 'aLanF9RyfpZowysgemsx'
                # }
                # files = {'media': open('static/predicted/ela_image.png', 'rb')}
                # r = requests.post('https://api.sightengine.com/1.0/check.json', files=files, data=params)

                # output = json.loads(r.text)
                # print(output)
                
                # client = SightengineClient("1641196078", "aLanF9RyfpZowysgemsx")
                # output = client.check('type').set_file('static/predicted/resaved_image.jpg')
                # print(output)
                
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
            else:
                data['error'] = "Invalid Format. upload JPEG/PNG Formats"
                res = render(request, 'base.html', data)
                return res
                # data = "Invalid Format. upload JPEG/PNG Formats"
                # render(request,  {'error':data})
        else:
            data['error'] = "please upload an image file"
            res = render(request, 'base.html', data)
            return res
                # data = "please upload an image file"
                # render(request, 'base.html', {'error':data})  

    return render(request, 'base.html')


                                                                                                                                                                            
def prepare_image(image_path):
    image_size = (128, 128)
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0         #normalizing the array values obtained from input image


# def prepare_img(image_path):
#     image_size = (128, 128)
#     return np.array(compute_ela_cv(image_path, 90).resize(image_size)).flatten() / 255.0


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

# def compute_ela_cv(path, quality):
#     temp_filename = 'temp_file_name.jpg'
#     SCALE = 15
#     orig_img = cv2.imread(path)
#     orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
#     cv2.imwrite(temp_filename, orig_img, [cv2.IMWRITE_JPEG_QUALITY, quality])

#     # read compressed image
#     compressed_img = cv2.imread(temp_filename)

#     # get absolute difference between img1 and img2 and multiply by scale
#     diff = SCALE * cv2.absdiff(orig_img, compressed_img)
#     diff.save("static/predicted/elacv/temp.png")
    
#     return diff