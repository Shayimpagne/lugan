import PIL.Image as Image
import numpy as np
from numpy import asarray
from numpy import savez_compressed
from numpy import load
import os

def convert_image_to_array(imageName):
    img = Image.open('dataset/{}'.format(imageName)).convert("L")
    imgArray = np.array(img)
    imgArray = (imgArray - 127.5) / 127.5
    return imgArray


def get_train_data():
    imagesArray = []

    for image in os.listdir('dataset'):
        if image.__contains__('.jpg'):
            imagesArray.append(convert_image_to_array(image))

    imagesArray = asarray(imagesArray)

    shape = imagesArray.shape
    imagesArray = imagesArray.reshape(shape[0], 16384)

    return imagesArray


def save_model():
    images = get_train_data()
    print(images.shape)
    filename = 'dataset_archive.npz'
    savez_compressed(filename, images)


def load_real_samples(filename):
    data = load(filename)
    images = data['arr_0']

    return images

