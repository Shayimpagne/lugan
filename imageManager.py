import cv2
import PIL.Image as Image
from matplotlib import pyplot
import random

def findFace(imageName):
    image = cv2.imread('downloads/{}'.format(imageName))
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    classifier = cv2.CascadeClassifier('/Users/emirshayymov/PycharmProjects/testTwo/venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    faces = classifier.detectMultiScale(grayImage, scaleFactor=1.1, minNeighbors=3)

    for (x, y, w, h) in faces:
        rectSide = w if w > h else h
        pt1 = (int(x), int(y))
        pt2 = (int(x + rectSide), int(y + rectSide))
        cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)

    print('founded {} faces'.format(len(faces)))

    size = (128, 128)

    for face in faces:
        pilImage = Image.open('downloads/{}'.format(imageName)).convert("RGB")
        img = pilImage.crop((face[0], face[1], face[0] + face[2], face[1] + face[3]))
        img = img.resize(size)
        img.save('dataset/{}_{}'.format(random.random(), imageName))
        print('transformed image' + imageName)

    print('--------')

