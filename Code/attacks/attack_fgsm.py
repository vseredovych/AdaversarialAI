import numpy as np
from numpy import linalg as LA

class AttackFGSM():
    # Iterative Fast Gradient Sign Method Attack
    def attack(self, X_vert, W, b, Y_vert_true, predict_func, max_norm=1, alpha=1):
        self.max_norm = max_norm
        self.alpha = alpha
        
        self.W = W
        self.b = b

        self.Y_ = np.expand_dims(Y_vert_true, axis=1)
        self.X_ = X_vert.T
        
        self.X_ = self.__gradient_descent(self.X_, self.Y_, self.W, self.b, predict_func)                
        return self.X_
    
    def __clip(self, X, X_adv, max_norm):
        max_res = np.maximum.reduce([np.zeros(X.shape), X - max_norm, X_adv])
        min_res = np.minimum.reduce([np.ones(X.shape), X + max_norm, max_res])
        return min_res

    
    def __softmax(self, z):
        exps = np.exp(z)
        return exps / exps.sum(axis=0, keepdims=True)
    
    def __stable_softmax(self, z):
        exps = np.exp(z - np.max(z))
        return exps / exps.sum(axis=0, keepdims=True)
        
    def __forward_backward_propagation(self, X, Y, W, b):
        # forward propagation
        m = X.shape[1]
        z = np.dot(W.T,X) + b
        A = self.__softmax(z)
        
        # backward propagation
        dz = A - Y
        derivative_x = (1 / m) * np.dot(W, dz)
        return np.array(derivative_x, dtype=float)

    def __is_tricked(self, X, Y, predict_func):
        # goal class position
        true_class = np.argmax(Y[:, 0] != 0)
        if (predict_func(np.array(X))[0] == true_class):
            return False
        return True

    def __gradient_descent(self, X, Y, W, b, predict_func):     
        m = X.shape[0]
        n = X.shape[1]
        
        delta_x = self.__forward_backward_propagation(X, Y, W, b)
        X_adv = X + self.alpha * np.sign(delta_x)
        X = self.__clip(X, X_adv, self.max_norm)
            

        if self.__is_tricked(X.T, Y, predict_func) == True:
            self.tricked = True
        else:
            self.tricked = False
        
        X_vert = X.T
        return X_vert