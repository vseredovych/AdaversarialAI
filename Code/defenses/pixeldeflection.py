import numpy as np
from random import randint, uniform

class PixelDeflection():
    def __init__(self, deflections=10, window=10):
        self.deflections = deflections
        self.window = window

    def get_defended(self, X, rcam_prob=None):
        if not rcam_prob is None:
            X_new = [
                self.pixel_deflection_with_map(x, rcam_prob, self.deflections, self.window) for x in X
            ] 
        else:
            X_new = [
                self.pixel_deflection_without_map(x, self.deflections, self.window) for x in X
            ]
            
        return X_new
    
    def pixel_deflection_without_map(self, img, deflections, window):
        np.random.seed(42)        
        img = np.copy(img)
        H, W = img.shape
        while deflections > 0:
                x,y = randint(0,H-1), randint(0,W-1)
                while True: #this is to ensure that PD pixel lies inside the image
                    a,b = randint(-1*window,window), randint(-1*window,window)
                    if x+a < H and x+a > 0 and y+b < W and y+b > 0: break
                # calling pixel deflection as pixel swap would be a misnomer,
                # as we can see below, it is one way copy
                img[x,y] = img[x+a,y+b] 
                deflections -= 1
        return img
    
    def pixel_deflection_with_map(self, img, rcam_prob, deflections, window):
        np.random.seed(42)        
        img = np.copy(img)
        H, W = img.shape
        while deflections > 0:
                x,y = randint(0,H-1), randint(0,W-1)
                # if a uniformly selected value is lower than the rcam probability
                # skip that region
                if uniform(0,1) < rcam_prob[x,y]:
                    continue

                while True: #this is to ensure that PD pixel lies inside the image
                    a,b = randint(-1*window,window), randint(-1*window,window)
                    if x+a < H and x+a > 0 and y+b < W and y+b > 0: break

                # calling pixel deflection as pixel swap would be a misnomer,
                # as we can see below, it is one way copy
                img[x,y] = img[x+a,y+b] 
                deflections -= 1
        return img