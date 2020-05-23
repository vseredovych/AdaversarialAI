import numpy as np
from numpy import linalg as LA
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

import numpy as np
from numpy import linalg as LA
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class CustomLogisticRegression():
    def __init__(self, 
                 normalize=True, 
                 learning_rate=0.1, 
                 num_iters=200, 
                 epsilon=1e-10, 
                 batch_size=1024, 
                 momentum=0.9):
        
        self.normalize = normalize
        self.__mean = None
        self.__std = None
        
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.momentum = momentum
        

    def get_mean(self):
        return self.__mean
    
    def get_std(self):
        return self.__std
    
    def fit(self, X_vert, Y_vert):
        # X transformations
        if self.normalize == True:
            self.X_ = self.__normalize(X_vert)
        else:
            self.X_ = X_vert
        self.X_ = self.X_.T
 
        # Y transformations
        self.Y_ = Y_vert.T

        self.W = np.full(( self.X_.shape[0],self.Y_.shape[0]),0.01)
        self.b = 0.0
        self.W, self.b, self.Js = self.__gradient_descent(
            self.X_, self.Y_, self.W, self.b, self.learning_rate, self.num_iters, self.epsilon, self.momentum, self.batch_size
        )

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)

        if self.normalize:
            X_norm = (X - self.__mean) / self.__std
            z = self.__stable_softmax(np.dot(self.W.T,X_norm.T)+self.b)
        else:
            z = self.__stable_softmax(np.dot(self.W.T,X.T)+self.b)
    
        y_pred = np.zeros(z.shape[1], dtype=int)
        for i in range(z.shape[1]):
            y_pred[i] = np.argmax(z[:, i])    
        
        return y_pred

    def predict_by_labels(self, X, y_labels):
        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        
        if self.normalize:
            X_norm = (X - self.__mean) / self.__std
            z = self.__stable_softmax(np.dot(self.W.T,X_norm.T)+self.b)
        else:
            z = self.__stable_softmax(np.dot(self.W.T,X.T)+self.b)
            
        y_pred = np.full((z.shape[1]), {})
        for i in range(z.shape[1]):
            y_pred[i] = { y_labels[j]: z[j][i] for j in range(z.shape[0])}
        return y_pred
            
    def get_cost_history(self):
        check_is_fitted(self)
        return self.Js

    def __normalize(self, X):
        if self.__mean is None and self.__std is None:
            mean = np.zeros([X.shape[1]])
            std  = np.ones([X.shape[1]])
            
            for i in range(X.shape[1]):
                if (np.std(X.iloc[:, i]) != 0):
                    mean[i] = np.mean(X.iloc[:, i])
                    std[i] = np.std(X.iloc[:, i])
            
            self.__mean = mean
            self.__std = std

        X_new = (X - self.__mean) / self.__std
        return X_new

    def __cost_function(self, X, Y, A):
        m = X.shape[0]
        if m == 0:
            return None
        
        J = (1 / m) * np.sum(-Y * np.log(A) - (1 - Y) * np.log(1 - A))
        return J

    def __cross_entropy(self, X, Y, A):
        m = X.shape[0]
        if m == 0:
            return None
        
        J = (-1 / m) * np.sum(Y.T * np.log(A.T))
        return J
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
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
        A = self.__stable_softmax(z)
        cost = self.__cross_entropy(X, Y, A)

        # backward propagation
        dz = A - Y
        derivative_weights = (1 / m) * np.dot(X, dz.T)
        derivative_bias = (1 / m) * np.sum(dz)

        return cost, derivative_weights, derivative_bias

    def __create_mini_batches(self, X, Y, batch_size):
        m = X.shape[1]

        if batch_size == -1:
            batch_size = m
                
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]        
        shuffled_Y = Y[:, permutation]

        mini_batches = []
        n_minibatches = m // batch_size
        
        for i in range(n_minibatches): 
            X_mini = shuffled_X[:, i * batch_size:(i + 1)*batch_size] 
            Y_mini = shuffled_Y[:, i * batch_size:(i + 1)*batch_size] 
            mini_batches.append((X_mini, Y_mini))
        if m % batch_size != 0:
            X_mini = shuffled_X[:, (i+1) * batch_size:m] 
            Y_mini = shuffled_Y[:, (i+1) * batch_size:m] 
            mini_batches.append((X_mini, Y_mini)) 
        return mini_batches
    
    def __gradient_descent(self, X, Y, W, b, learning_rate, num_iters, epsilon, momentum, batch_size):        
        # num of samples
        print(self.get_params())
        # num of features
        n = X.shape[1]

        J_history = []
        VdW = 0
        Vdb = 0
        for i in range(num_iters):
            mini_batches = self.__create_mini_batches(X, Y, batch_size) 
            for (X_mini, Y_mini) in mini_batches:
                J, delta_weights, delta_bias = self.__forward_backward_propagation(X_mini, Y_mini, W, b)
                
                VdW = (momentum * VdW + (1 - momentum) * delta_weights / (1 - momentum ** (i + 1)))
                Vdb = (momentum * Vdb + (1 - momentum) * delta_bias / (1 - momentum ** (i + 1)))
                
                W = W - learning_rate * VdW
                b = b - learning_rate * Vdb

            if i % 100 == 0:
                print(f"{i} iteration: {J}")

            J_history.append(J)

        return W, b, J_history
    
    def get_params(self, deep=True):
        return {
                "learning_rate": self.learning_rate,
                "num_iters": self.num_iters,
                "batch_size": self.batch_size,
                "momentum": self.momentum,
                "normalize": self.normalize
               }
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self