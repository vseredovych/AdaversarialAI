import numpy as np
from numpy import linalg as LA

class AttackTIFGSM():    
    # Targeted Iterative Fast Gradient Sign Method Attack
    def attack(self, X_vert, W, b, Y_vert_goal, predict_func, max_norm=1, alpha=0.01, max_iters=100):
        self.alpha = alpha
        self.num_iters = 0
        self.max_iters = max_iters
        self.max_norm = max_norm
        
        self.W = W
        self.b = b

        self.Y_ = np.expand_dims(Y_vert_goal, axis=1)
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
        
    def __forward_backward_propagation(self, X, Y, W, b):
        m = X.shape[1]
        z = np.dot(W.T,X) + b
        A = self.__softmax(z)

        dz = A - Y
        derivative_x = np.dot(W, dz)
        return derivative_x

    def __is_tricked(self, X, Y, predict_func):
        goal_class = np.argmax(Y[:, 0] != 0)
        if (predict_func(np.array(X))[0] != goal_class):
            return False
        return True
    
    def __gradient_descent(self, X, Y, W, b, predict_func):        
        # num of features
        m = X.shape[0]
        # num of samples
        n = X.shape[1]
        
        X_origin = X.copy()
        #alpha = self.max_norm / self.max_iters
        
        for i in range(self.max_iters):
            delta_x = self.__forward_backward_propagation(X, Y, W, b)
            X_adv = X - self.alpha * np.sign(delta_x)
            X = self.__clip(X_origin, X_adv, self.max_norm)

            self.num_iters += 1
                
            if self.__is_tricked(X.T, Y, predict_func) == True:
                self.tricked = True
                break
            else:
                self.tricked = False
        
        X_vert = X.T
        return X_vert