import numpy as np

def chw2hwc(img):
    return np.transpose(a=img, axes=(1,2,0))

def naive_normalize(x):
    return (x - np.amin(x)) / (np.amax(x) - np.amin(x))