import requests
import shutil
from PIL.Image import Image

def downloadFile(imageUrl, imageName):
    # filename = imageUrl.split("/")[-1]

    # Open the url image, set stream to True, this will return the stream content.
    r = requests.get(imageUrl, stream=True)

    # Check if the image was retrieved successfully
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True

        # Open a local file with wb ( write binary ) permission.
        with open('downloads/{}.jpg'.format(imageName), 'wb') as f:
            shutil.copyfileobj(r.raw, f)

        print('Image sucessfully Downloaded: ', imageName)
    else:
        print('Image Couldn\'t be retreived')

def downloadAllImages(csvFileName):

    count = 0

    for line in open(csvFileName, 'r').readlines():
        downloadFile(line.strip(), '{}'.format(count))
        count += 1

def resizeImage(image, folder, size, imageName):
    originalImage = Image.open('{}{}'.format(folder, image)).convert("RGB")
    w, h = originalImage.size
    minimumSide = w if w < h else h
    img = (originalImage.crop((0, 0, minimumSide, minimumSide))).resize((size, size))

    img.save('originalImages/' + imageName)