import numpy as np
from random import randint, uniform

class Randomization():
    def __init__(self, resize_window=5, pad_window=5, w=28, h=28):
        self.default_w = w
        self.default_h = h
        self.resize_window = resize_window
        self.pad_window = pad_window
    
    def get_defended(self, X):
        return self.__random_resize_paddding(X)
    
    def __random_resize_paddding(self, X):
        X_new = np.zeros(X.shape)

        for idx, x in enumerate(X):
            default_w, default_h = (28, 28)

            w, h = np.random.randint(-self.resize_window, self.resize_window, 2)
            if self.pad_window == 0:
                p_t, p_b, p_l, p_r = (0, 0, 0, 0)
            else:
                p_t, p_b, p_l, p_r = np.random.randint(0, self.pad_window, 4)
            
            x_scaled = self.__scale(x, default_w + w, default_h + h)
            x_scaled_padded = np.pad(x_scaled, ((p_t, p_b), (p_l, p_r)), 'constant')
            x_normal = self.__scale(x_scaled_padded, default_w, default_h)

            X_new[idx] = x_normal    
        return X_new

    def __scale(self, img, n_rows, n_cols):
        n_rows0 = img.shape[0]
        n_cols0 = img.shape[1]
        new_img = np.zeros((n_rows, n_cols))

        r_idxs = (n_rows0 * np.arange(n_rows) / n_rows).astype(int)
        c_idxs = (n_cols0 * np.arange(n_cols) / n_cols).astype(int)
        for i, r_idx in enumerate(r_idxs):
            for j, c_idx in enumerate(c_idxs):
                new_img[i][j] = img[r_idx][c_idx]

        return new_img