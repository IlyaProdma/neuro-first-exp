from PIL import Image
import numpy as np

def create():

    results = [0, 0, 1, 1, 1, 1, 0, 1, 1,
               1, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 1, 0, 0, 1, 0, 0, 0, 0,
               0, 0, 0, 0, 1, 1, 0, 1, 0,
               1, 1, 0, 0, 0, 1, 1, 1, 1,
               0, 0, 1, 1, 0, 0, 0, 0, 0,
               1, 0, 1]
    f_out = open('dataset.txt', 'w', encoding='utf-8')
    for i in range(1,58):
        filename = "raw_photos/"+str(i)+".jpg"
        im = Image.open(filename)
        width, height = im.size
        newsize = (16, 16)
        im1 = im.resize(newsize).convert('L')
        data = np.asarray(im1)
        for row in data:
            for pixel in row:
                f_out.write(str(pixel) + ' ')
        f_out.write(str(results[i-1]) + '\n')
    f_out.close()

def read():
    f_in = open('dataset.txt', 'r')
    for line in f_in.readlines():
        print(len(line.split()))

def image_to_array(filename: str):
    result = []
    im1 = Image.open(filename).resize((16,16)).convert('L')
    data = np.asarray(im1)
    for row in data:
        for pixel in row:
            result.append(pixel)
    return result
