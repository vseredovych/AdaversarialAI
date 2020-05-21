import numpy as np
from numpy import linalg as LA

class AttackDeepFool():
    # Deep Fool Attack    
    def attack(self, X_vert, W, b, Y_vert_true, predict_func, max_norm=12, alpha=1, max_iters=100):
        self.num_iters = 0
        self.max_iters = max_iters
        self.max_norm = max_norm
        self.alpha = alpha
        
        self.W = W
        self.b = b

        self.Y_ = np.expand_dims(Y_vert_true, axis=1)
        self.X_ = X_vert.T
        
        self.X_ = self.__deep_fool(self.X_, self.Y_, self.W, self.b, predict_func)                
        return self.X_
    
    def __clip(self, X, X_adv, max_norm):
        max_res = np.maximum.reduce([np.zeros(X.shape), X - max_norm, X_adv])
        min_res = np.minimum.reduce([np.ones(X.shape), X + max_norm, max_res])
        return min_res
    
    def __stable_softmax(self, z):
        exps = np.exp(z - np.max(z))
        return exps / exps.sum(axis=0, keepdims=True)
        
    def __softmax(self, z):
        exps = np.exp(z)
        return exps / exps.sum(axis=0, keepdims=True)

    def __is_tricked(self, X, Y, predict_func):
        true_class = np.argmax(Y[:, 0] != 0)
        if (predict_func(np.array(X))[0] == true_class):
            return False
        return True

    def __classificator(self, W, b, X):
        return self.__softmax(np.dot(W.T,X)+ b)[:, 0]
    
    def __classificator_gradient(self, W, b, X, Y, k):
        a = self.__classificator(W, b, X)
        return a[k] * (1 - a[k]) * W.T[k]
    
    def __deep_fool(self, X, Y, W, b, predict_func):        
        # num of features
        m = X.shape[0]
        # num of samples
        n = X.shape[1]
        
        X_origin = X.copy()
        #alpha = self.max_norm / self.max_iters
        idxs = [x[0] for x in np.argwhere(Y[:, 0] == 0)]
        true_class = np.argwhere(Y[:, 0] != 0)[0]

        while (LA.norm((X_origin - X), np.inf) < self.max_norm and
               self.__is_tricked(X.T, Y, predict_func) == False and
               self.num_iters < self.max_iters):
            
            w_gradients = []
            f_predictions = []

            grad_origin = self.__classificator_gradient(W, b, X, Y, true_class)
            pred_origin = self.__classificator(W, b, X)[true_class]

            for k in idxs:
                grad_k = self.__classificator_gradient(W, b, X, Y, k)
                pred_k = self.__classificator(W, b, X)[k]
                
                w_k = grad_origin - grad_k
                f_k = pred_origin - pred_k
            
                w_gradients.append(grad_k)
                f_predictions.append(f_k)
            
            w_gradients_norm = [LA.norm(w, 2) for w in w_gradients]
            l = np.argmin( np.abs(np.array(f_k)) / w_gradients_norm )
            
            perturbation = (f_predictions[l] * w_gradients[l]) / w_gradients_norm[l] ** 2
            
            X_adv = X + self.alpha * np.expand_dims(perturbation, axis=1)
            X = self.__clip(X, X_adv, self.max_norm)
            
            self.num_iters += 1
            #print(LA.norm(X_origin - X, np.inf))

        if self.__is_tricked(X.T, Y, predict_func) == True:
            self.tricked = True
        else:
            self.tricked = False
        #print(self.tricked)
        X_vert = X.T
        return X_vert
    
    
    
# import numpy as np
# from numpy import linalg as LA

# class AttackDeepFool():
#     # Deep Fool Attack    
#     def attack(self, X_vert, W, b, Y_vert_true, predict_func, max_norm=12, alpha=1, max_iters=100):
#         self.num_iters = 0
#         self.max_iters = max_iters
#         self.max_norm = max_norm
#         self.alpha = alpha
        
#         self.W = W
#         self.b = b

#         self.Y_ = np.expand_dims(Y_vert_true, axis=1)
#         self.X_ = X_vert.T
        
#         self.X_ = self.__deep_fool(self.X_, self.Y_, self.W, self.b, predict_func)                
#         return self.X_
    
#     def __clip(self, X, X_adv, max_norm):
#         max_res = np.maximum.reduce([np.zeros(X.shape), X - max_norm, X_adv])
#         min_res = np.minimum.reduce([np.ones(X.shape), X + max_norm, max_res])
#         return min_res

    
#     def __stable_softmax(self, z):
#         exps = np.exp(z - np.max(z))
#         return exps / exps.sum(axis=0, keepdims=True)
        
#     def __softmax(self, z):
#         exps = np.exp(z)
#         return exps / exps.sum(axis=0, keepdims=True)

#     def __is_tricked(self, X, Y, predict_func):
#         true_class = np.argmax(Y[:, 0] != 0)
#         if (predict_func(np.array(X))[0] == true_class):
#             return False
#         return True

#     def __classificator(self, W, b, X):
#         return self.__softmax(np.dot(W.T,X)+ b)[:, 0]
    
#     def __classificator_gradient(self, W, b, X, Y, k):
#         a = self.__classificator(W, b, X)
#         return a[k] * (1 - a[k]) * W.T[k]
    
#     def __deep_fool(self, X, Y, W, b, predict_func):        
#         # num of features
#         m = X.shape[0]
#         # num of samples
#         n = X.shape[1]
        
#         X_origin = X.copy()
#         #alpha = self.max_norm / self.max_iters
#         idxs = [x[0] for x in np.argwhere(Y[:, 0] == 0)]
#         true_class = np.argwhere(Y[:, 0] != 0)[0]

#         while (LA.norm((X_origin - X), np.inf) < self.max_norm and
#                self.__is_tricked(X.T, Y, predict_func) == False and
#                self.num_iters < self.max_iters):
            
#             w_gradients = []
#             f_predictions = []

#             grad_origin = self.__classificator_gradient(W, b, X, Y, true_class)
#             pred_origin = self.__classificator(W, b, X)[true_class]

#             for k in idxs:
#                 grad_k = self.__classificator_gradient(W, b, X, Y, k)
#                 pred_k = self.__classificator(W, b, X)[k]
                
#                 w_k = grad_origin - grad_k
#                 f_k = pred_origin - pred_k
            
#                 w_gradients.append(grad_k)
#                 f_predictions.append(f_k)
            
#             w_gradients_norm = [LA.norm(w, 1) for w in w_gradients]
#             l = np.argmin( np.abs(np.array(f_k)) / w_gradients_norm )
            
#             perturbation = (f_predictions[l] * np.sign(w_gradients[l])) / w_gradients_norm[l]
            
#             X_adv = X + self.alpha * np.expand_dims(perturbation, axis=1)
#             X = self.__clip(X, X_adv, self.max_norm)
            
#             self.num_iters += 1
#             #print(LA.norm(X_origin - X, np.inf))

#         if self.__is_tricked(X.T, Y, predict_func) == True:
#             self.tricked = True
#         else:
#             self.tricked = False
#         #print(self.tricked)
#         X_vert = X.T
#         return X_vert