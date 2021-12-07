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
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error
#from statsmodels.stats.contingency_tables import mcnemar
from mlxtend.evaluate import mcnemar



def masked_non_zeros(data):
    masked =np.all(data ==0, axis= 0 )
    amount_of_zeros = np.sum(masked)
    masked = np.where(masked==True,False, True)
    masked_digit = data[:, masked]

    return (amount_of_zeros, masked_digit)

def model_get_best_grid_params(model):
    name_model =str(model)[:-2]
    if name_model =="MLPClassifier":
        return (name_model, {'alpha': [0.01, 0.1, 0.25, 0.5, 1, 2.5, 10]})
    elif name_model =="LogisticRegressionCV":
        return (name_model, {"l1_ratios":[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]})
    elif name_model =="LogisticRegression":
        return( name_model, {'penalty': ['l1'], 'C': [0.01, 0.1, 0.25, 0.5, 1, 2.5, 10]})
    elif name_model == "SVC":
        return (name_model, {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']})



def ncnemars_test(y,model_1,model_2):
    
    output_matrix = np.zeros((2,2)) #creating the output matrix
    #check when model are inacuurate and place it in the right loc in the matrix
    output_matrix[1][0] = np.sum(np.where((y ==model_1 ) & (y!=model_2),1,0))  #model 1 TN or TP, modele 2 FN or FP
    output_matrix[0][1] = np.sum(np.where(((y !=model_1) & (y==model_2) ),1,0))  #model 2 TN or TP, modele 1 FN or FP
    
    #check when model are are the same in both ==> their correct output and mistakes
    output_matrix[0][0] = np.sum(np.where((y!= model_2 ) & (y!= model_1 ),1,0)) #both models incorrect (FN or FP)
    output_matrix[1][1] =np.sum(np.where((y==model_1) & (y==model_2) ,1,0)) #both models correct (TP or TN)
    
    
    
    #check if we can divide
    if (output_matrix[1][0] + output_matrix[0][1] != 0):
        p_value = (((abs(output_matrix[1][0] - (output_matrix[0][1])))-1)**2 )/ ((output_matrix[1][0] + output_matrix[0][1]))
    # if not
    else:
        p_value  ="can not divided by 0: The false positive and false negative are equal"
        
    return(output_matrix,p_value)



arr_ = np.array([[1,2,3,4,1,2,3,1]])
arr_2 = np.array([[1,2,3,3,1,1,3,4]])

y = np.array([[1,2,3,3,4,4,2,1]])
assert y.shape ==arr_.shape ==arr_2.shape

ncnemars_test(y, arr_, arr_2)
mnist_data = pd.read_csv('mnist.csv').values

labels = mnist_data[:, 0] # ==> <class 'numpy.ndarray'> original R 1x784
digits = mnist_data[:, 1:] # ==> <class 'numpy.ndarray'>
img_size = 28



# Remove Features with only 0 values ==> Not a must since it will be taken care at a later stage as well
(amount_of_zeros, non_zero_df) = masked_non_zeros(digits)

print(f" Amount of Zeros {amount_of_zeros} ; Non Zero Shapes: {non_zero_df.shape} ; Digit_Shpes {digits.shape}")
#Standardize The data
standardized_data= StandardScaler().fit_transform(non_zero_df)



pca_3 = decomposition.PCA()
pca_3.n_components = 196
pca_data_19 = pca_3.fit_transform(standardized_data)
print(pca_data_19.shape)

#name accuracy test, accuracy train, precision, recall F1 Score, values
values = []
confusion_matrixes = []

df = pd.DataFrame(data = None, columns=("name" ,"accuracy_test", "accuracy train", "precision", "recall" ,"F1 Score", ))
df_2 = pd.DataFrame(data = None, columns=("name" , "accuracy_test", "accuracy train", "precision", "recall" ,"F1 Score",))

X_train,X_test,y_train,Y_test = train_test_split(pca_data_19, labels,shuffle= 40, train_size=0.2)
idx = 0

save = False
predictions_list  = []

print(Y_test.shape)
for model in [ LogisticRegression(), LogisticRegressionCV(),SVC(),MLPClassifier()]: #LogisticRegression(),
    (name ,params) = model_get_best_grid_params(str(model))
    if name =='':
        model = LogisticRegression(solver = 'lbfgs')
    else: 

        grid_model =GridSearchCV(model, params, cv=5)
        
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    
    predictions_list.append((name, predicted))
    confusion_matrix = metrics.confusion_matrix(predicted, Y_test)
    confusion_matrixes.append(((name, confusion_matrix)))
    
    
    training_score = model.score(X_train, y_train)
    expenend_ = np.expand_dims(predicted, axis = 0)
    test_score = model.score(X_test, Y_test)
    
    
    precision, recall, fscore, _ = precision_recall_fscore_support(predicted, Y_test, average = 'macro')
    result = [name,test_score, training_score,precision, recall, fscore]
    
    df_2.loc[len(result)] = result
    df = pd.concat([df,df_2]).reset_index(drop = True)
    
    idx +=1
    
saved_p_val =True
df_p = pd.DataFrame(columns=("Model_1", "Model_2", "P_Value"))
df_2P = pd.DataFrame(columns=("Model_1", "Model_2", "P_Value"))
output_matrixes_p_val = []

tables = []
for prediction_1 in predictions_list:
    model_1, prediction_1 = prediction_1
    for prediction_2 in predictions_list:
        model_2, prediction_2 = prediction_2
        if (model_1 ==model_2):
            pass
        else:
            output_matrix, p_value= ncnemars_test(Y_test,prediction_1, prediction_2)
            
            chi2, p = mcnemar(ary=output_matrix, corrected=True)

            print(f"{model_1} {model_2} Chai: {chi2}, P: {p}")
           # tables.append((model_1,model_2,table,p_val, static))

            result_p_val = [model_1, model_2, p_value]
            df_2P.loc[len(result_p_val)] = result_p_val
            df_p = pd.concat([df_p, df_2P]).reset_index(drop = True)
            output_matrixes_p_val.append((model_1,model_2, output_matrix))

if saved_p_val:
    df_p.to_csv("Pvalus_models_")
    with open("P_value_matrixes",'w') as p_val_matrix:
        for (model_1, model_2, output_mat) in output_matrixes_p_val:
            p_val_matrix.write(f"models: {str(model_1)} , {str(model_2)}" + '\n\n')
            p_val_matrix.write(str(output_mat))
    

if save: 
    df.to_csv("Models.csv", index = False)
    with open("Confusion_Matrixes.txt", 'w') as output:       
        for mat in confusion_matrixes:
            (name_mat , matrix_) = mat
            output.write(str(name_mat) + '\n\n')
            output.write(str(matrix_) + '\n\n')
            
            
  
    