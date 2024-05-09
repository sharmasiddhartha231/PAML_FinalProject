import pandas as pd
import numpy as np

class LogisticRegression1(object):
    def __init__(self, learning_rate=0.001, num_iterations=1000): 
        self.learning_rate = learning_rate 
        self.num_iterations = num_iterations 
        self.likelihood_history=[]
    def predict_probability(self, X):
        score = np.dot(X, self.W) + self.b
        y_pred = 1. / (1.+np.exp(-score)) 
        return y_pred
    def compute_avg_log_likelihood(self, X, Y, W):
        #indicator = (Y==+1)
        #scores = np.dot(X, W) 
        #logexp = np.log(1. + np.exp(-scores))
        #mask = np.isinf(logexp)
        #logexp[mask] = -scores[mask]
        #lp = np.sum((indicator-1)*scores - logexp)/len(X)
        scores = np.dot(X, W) 
        score = 1 / (1 + np.exp(-scores))
        y1 = (Y * np.log(score))
        y2 = (1-Y) * np.log(1 - score)
        lp = -np.mean(y1 + y2)
        return lp
    def update_weights(self):      
        num_examples, num_features = self.X.shape
        y_pred = self.predict(self.X)
        dW = self.X.T.dot(self.Y-y_pred) / num_examples 
        db = np.sum(self.Y-y_pred) / num_examples 
        self.b = self.b + self.learning_rate * db
        self.W = self.W + self.learning_rate * dW
        log_likelihood=0
        log_likelihood += self.compute_avg_log_likelihood(self.X, self.Y, self.W)
        self.likelihood_history.append(log_likelihood)
    def predict(self, X):
        y_pred= 0
        scores = 1 / (1 + np.exp(- (X.dot(self.W) + self.b)))
        y_pred = [0 if z <= 0.5 else +1 for z in scores]
        return y_pred 
    def fit(self, X, Y):   
        self.X = X
        self.Y = Y
        num_examples, num_features = self.X.shape    
        self.W = np.zeros(num_features)
        self.b = 0
        self.likelihood_history=[]
        for _ in range(self.num_iterations):          
            self.update_weights()  
    def get_weights(self):
            out_dict = {'Logistic Regression': []}
            W = np.array([f for f in self.W])
            out_dict['Logistic Regression'] = self.W
            return out_dict

class LogisticRegression(object):
    def __init__(self, learning_rate=0.001, num_iterations=1000): 
        self.learning_rate = learning_rate 
        self.num_iterations = num_iterations 
        self.likelihood_history=[]
    def predict_probability(self, X):
        score = np.dot(X, self.W) + self.b
        y_pred = 1. / (1.+np.exp(-score)) 
        return y_pred
    def compute_avg_log_likelihood(self, X, Y, W):
        indicator = (Y==+1)
        scores = np.dot(X, W) 
        logexp = np.log(1. + np.exp(-scores))
        mask = np.isinf(logexp)
        logexp[mask] = -scores[mask]
        lp = np.sum((indicator-1)*scores - logexp)/len(X)
        return lp
    def update_weights(self):      
        num_examples, num_features = self.X.shape
        y_pred = self.predict(self.X)
        dW = self.X.T.dot(self.Y-y_pred) / num_examples 
        db = np.sum(self.Y-y_pred) / num_examples 
        self.b = self.b + self.learning_rate * db
        self.W = self.W + self.learning_rate * dW
        log_likelihood=0
        log_likelihood += self.compute_avg_log_likelihood(self.X, self.Y, self.W)
        self.likelihood_history.append(log_likelihood)
    def predict(self, X):
        y_pred= 0
        scores = 1 / (1 + np.exp(- (X.dot(self.W) + self.b)))
        y_pred = [0 if z <= 0.5 else +1 for z in scores]
        return y_pred 
    def fit(self, X, Y):   
        self.X = X
        self.Y = Y
        num_examples, num_features = self.X.shape    
        self.W = np.zeros(num_features)
        self.b = 0
        self.likelihood_history=[]
        for _ in range(self.num_iterations):          
            self.update_weights()  
    def get_weights(self):
            out_dict = {'Logistic Regression': []}
            W = np.array([f for f in self.W])
            out_dict['Logistic Regression'] = self.W
            return out_dict




lg_model = LogisticRegression(num_iterations=1000, learning_rate=0.001)
lg_model1 = LogisticRegression1(num_iterations=1000, learning_rate=0.001)

lg_model.fit(X_train, y_train)
lg_model1.fit(X_train, y_train) 
              
y_pred1 = lg_model1.predict(X_test)
y_pred = lg_model.predict(X_test)

c1 = confusion_matrix(y_test, y_pred1, labels=[0,1])
a1 = lg_model1.get_weights()

c= confusion_matrix(y_test, y_pred, labels=[0,1])
a = lg_model.get_weights()

tn, fp, fn, tp = c1.ravel()
specificity = tn / (tn+fp)
accuracy = (tp + tn)/(tp + tn + fp + fn)
misclassification = (fp + fn)/(tp + tn + fp + fn)
sensitivity = tp /(tp + fn)
precision = tp / (tp + fp)
accuracy
sensitivity
precision
specificity
a1
