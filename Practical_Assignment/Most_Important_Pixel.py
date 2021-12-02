
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LogisticRegression,  LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from cv2 import resize
from scipy.linalg import eigh # eigen value and eigen vector for dimentally reduction 
import seaborn as sns
from sklearn.model_selection import train_test_split
import csv
from io import StringIO   # StringIO behaves like a file object



def model_get_best_grid_params(model):
    name_model =str(model)[:-2]
    if name_model =="MLPClassifier":
        return (name_model, {'alpha': [0.01, 0.1, 0.25, 0.5, 1, 2.5, 10]})
    elif name_model =="LogisticRegressionCV":
        return (name_model, {"l1_ratios":[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]})
    elif name_model =="LogisticRegression":
        return( name_model, {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 0.25, 0.5, 1, 2.5, 10]})
    elif name_model == "SVC":
        return (name_model, {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']})


def pixel_by_digit(labels):
    
    array_pixel_by_digit = [[0] for i in range(10)]
    num_occur = [[0] for i in range(10)]

    for idx, num in enumerate(labels):
        curr_pixel_row = np.expand_dims(digits[idx,:],axis=0) 
        num_array_array = array_pixel_by_digit[num]
        num_occur[num][0] +=1
        
        if (type(num_array_array[0])==int):
            array_pixel_by_digit[num] =curr_pixel_row
        else: 
            passed_rows = num_array_array
            concat_mat = np.concatenate((curr_pixel_row, passed_rows), axis = 0)
            array_pixel_by_digit[num] = concat_mat
    return pixel_by_digit



mnist_data = pd.read_csv('mnist.csv').values

labels = mnist_data[:, 0] # ==> <class 'numpy.ndarray'> original R 1x784
digits = mnist_data[:, 1:] # ==> <class 'numpy.ndarray'>
img_size = 28



sum_intensity = np.sum(digits,axis= 0)
most_frequqnt_pixel = np.argmax(sum_intensity)


array_pixel_by_digit = np.loadtxt('Num_Pixel_By_Digit.csv')


ys = []
y = []
x = list([np.arange(10)])

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30,10), squeeze=False)

for n , num_features in zip (range(3),np.arange(10,180,40)) :

    for num in range(10):
        inx_max = np.argmax(np.average(array_pixel_by_digit[num], axis = 0))
        importance_pixel  = np.argsort(np.average(array_pixel_by_digit[num], axis = 0))[-num_features::]
        y.append(importance_pixel)
    
    ys.append(y)
    
    
    


concat = np.expand_dims(ys[1][1], axis = 0)
for i in range(1,10):
    concat  = np.concatenate(   (concat, np.expand_dims(ys[1][i],axis = 0)) , axis =0)
    
x,X_test,y,Y_test = train_test_split(concat, np.arange(10), shuffle=6)

    
for k in [2,10]:
    for model in [ LogisticRegression()]:
        name, params = model_get_best_grid_params(model)
        
        # (Accuracy, Precision, Recall, F-score)
        tuned_params = GridSearchCV(model, params, cv=k, n_jobs=-1)
        tuned_params.fit(x, y)
        y_pred = tuned_params.predict(X_test)
