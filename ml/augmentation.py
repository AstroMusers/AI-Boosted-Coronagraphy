import numpy as np

class Augmentation():
    def __init__(self):
        print("Augmentation class init")
    
    def normalize(self, img):
        print("Normalization")
        return (img - img.min()) / (img.max() - img.min())

    def rotate90(self, img, times=1):
        print("Counterclockwise rotation")
        return np.rot90(img, k=times)
    
    def flip(self, img, horizontal=True, vertical=True):
        print("Flip")
        return np.flipud(np.fliplr(img)) if horizontal and vertical else (np.flipud(img) if vertical else (np.fliplr(img) if horizontal else img))

    def shift(self, img, right_shift, down_shift):
        print("Shift")
        return np.roll(np.roll(img, right_shift, axis=1), down_shift, axis=0)