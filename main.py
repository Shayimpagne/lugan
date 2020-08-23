import downloader
import imageManager
import loader
import GAN
import os
from numpy import savez_compressed

# downloader.downloadAllImages('urls.csv')

"""
for image in os.listdir('downloads'):
    if image.__contains__('.jpg'):
        print('for image' + image)
        imageManager.findFace(image)
"""

#loader.save_model()

images = loader.load_real_samples('dataset_archive.npz')
print(images.shape)

GAN.train(epochs=100, batch_size=128, trainData=images)

#GAN.generate_image()