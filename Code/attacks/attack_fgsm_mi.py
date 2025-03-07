import numpy as np
from numpy import linalg as LA

class AttackMIFGSM():
    # Iterative Fast Gradient Sign Method Attack
    def attack(self, X_vert, W, b, Y_vert_true, predict_func, max_norm=0.1, alpha=0.01, momentum=0.9, max_iters=100):
        self.num_iters = 0
        self.max_iters = max_iters
        self.max_norm = max_norm
        self.alpha = alpha
        self.momentum = momentum
        
        
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
   
    def __softmax_bigfloat(self, z):
        z = np.array(z, dtype=Float)
        exps = Float(np.e) ** z
        return exps / exps.sum(axis=0, keepdims=True)
    
    def __forward_backward_propagation(self, X, Y, W, b):
        m = X.shape[1]
        z = np.dot(W.T,X) + b
        A = self.__softmax(z)
        
        dz = A - Y
        derivative_x = (1 / m) * np.dot(W, dz)
        return np.array(derivative_x, dtype=float)

    def __is_tricked(self, X, Y, predict_func):
        true_class = np.argmax(Y[:, 0] != 0)
        if (predict_func(np.array(X))[0] == true_class):
            return False
        return True

    def __gradient_descent(self, X, Y, W, b, predict_func):        
        # num of features
        m = X.shape[0]
        # num of samples
        n = X.shape[1]
        
        X_origin = X.copy()
        #alpha = self.max_norm / self.max_iters
        g0 = 0
        
        for i in range(self.max_iters):
            delta_x = self.__forward_backward_propagation(X, Y, W, b)
            g1 = self.momentum * g0 + delta_x
            g0 = g1

            X_adv = X + self.alpha * np.sign(g1)
            X = self.__clip(X_origin, X_adv, self.max_norm)
            
            self.num_iters += 1
                
            if self.__is_tricked(X.T, Y, predict_func) == True:
                self.tricked = True
                break
            else:
                self.tricked = False
                
        X_vert = X.T
        return X_vert