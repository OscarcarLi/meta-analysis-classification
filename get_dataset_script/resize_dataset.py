import os
import sys
import numpy as np
from imageio import imread, imwrite
from skimage.transform import resize

target_dir = './data/aircraft' if len(sys.argv) < 2 else sys.argv[1]
img_size = [84, 84]

_ids = []

for root, dirnames, filenames in os.walk(target_dir):
    for filename in filenames:
        if filename.endswith(('.jpg', '.webp', '.JPEG', '.png', 'jpeg')):
            _ids.append(os.path.join(root, filename))

for i, path in enumerate(_ids):
    img = imread(path)
    print('{}/{} size: {}'.format(i, len(_ids), img.shape))
    resize_img = (255 * resize(img, img_size)).astype(np.uint8)
    imwrite(path, resize_img)
