import numpy as np
from numpy import linalg as LA

class AttackNoise():
    def __clip(self, X, X_adv, max_norm):
        max_res = np.maximum.reduce([np.zeros(X.shape), X - max_norm, X_adv])
        min_res = np.minimum.reduce([np.ones(X.shape) * 1, X + max_norm, max_res])
        return min_res
    
    # Targeted Fast Gradient Sign Method Attack
    def attack(self, X_vert, Y_vert_true, predict_func, max_norm=1):
        self.max_norm = max_norm

        self.X_ = X_vert.T
        self.Y_ = np.expand_dims(Y_vert_true, axis=1)
        
        self.X_ = self.__add_noise(self.X_, self.Y_, predict_func)                
        return self.X_
   
    def __generate_noise(self, m, n):
        return np.random.uniform(low=-1, high=1, size=(m, n)) * self.max_norm

    def __is_tricked(self, X, Y, predict_func):
        # true class position
        true_class = np.argmax(Y[:, 0] != 0)

        if (predict_func(np.array(X))[0] == true_class):
            return False
        return True
    
    def __add_noise(self, X, Y, predict_func):
        # num of features
        m = X.shape[0]
        # num of samples
        n = X.shape[1]

        noise = self.__generate_noise(m, n);
        
        X_adv = X + noise
        X = self.__clip(X, X_adv, self.max_norm)
       
        if self.__is_tricked(X.T, Y, predict_func) == True:
            self.tricked = True
        else:
            self.tricked = False
        
        X_vert = X.T
        return X_vert
