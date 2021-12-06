
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
import scipy.stats as stats



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

not_zero = np.sum(np.where(digits != 0, 1, 0))
np_zero = np.sum(np.where(digits == 0, 1, 0))

tatio =  (np_zero /(not_zero + np_zero))



sum_intensity = np.sum(digits,axis= 0)
most_frequqnt_pixel = np.argmax(sum_intensity)



array_pixel_by_digit = [[0] for i in range(10)]
num_occur = [[0] for i in range(10)]


sorted_by_digi = mnist_data[mnist_data[:, 0].argsort()]
num_features =50

How_Many_digits_For_class =[sorted_by_digi[sorted_by_digi[:,0] ==x].shape[0] for x in range(10)]

std_diti = np.std(How_Many_digits_For_class)
mean_diti = np.mean(How_Many_digits_For_class)
variance = np.var(How_Many_digits_For_class)
media  = np.median(How_Many_digits_For_class)

g = sns.distplot(sorted(How_Many_digits_For_class)) #, kde = False
#g.set_yticks(range(0,10,2))
g.set_xlabel("Values")


most_im = [np.argsort(np.average(sorted_by_digi[sorted_by_digi[:,0]==idx], axis = 0))[-num_features::] for idx in range(10)]
most_important_pixels = [sorted_by_digi[sorted_by_digi[:,0]== idx,:] for idx, col in enumerate(most_im)  ]


concat =0
check_not_in = []

for idx, row in enumerate(most_im):
    not_included = [x for x in range(1,784) if x not in list(row)]
    check_not_in.append(not_included)
    masked = most_important_pixels[idx]
    sored = np.sort(row)
        #masked[:,list(list_loc)]= np.array(np.max(masked)) #           np.ones(1)  #
    print("masked_1")
    print(masked)
    masked[:,not_included]=np.zeros(1)
    print(masked)
    print("masked_2")
    print(idx)
    masked[:,0] = np.array(idx)

    if isinstance(concat, np.ndarray):
        concat = np.concatenate((concat,masked), axis = 0)
        
        print("concat")
    else: 
        concat = masked

np.random.shuffle(concat)


model= LogisticRegressionCV()
model.fit(concat[:,1:],concat[:,0] )

predict = model.predict(mnist_data[:5000, 1:])
accuracy =(np.sum(predict ==mnist_data[:5000, 0]))/ 5000

print(predict)