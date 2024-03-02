import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data_NaN = pd.read_csv("heart-disease-dataV2.csv") # FIX THIS to better relative 
data = data_NaN.dropna().reset_index(drop=True) # cleared NaN values and reset id's 
print(data)

y=data.HeartDisease #strength is the label we want to predict
x=data.drop('HeartDisease',axis=1)#use drop function to take all other data in x

print(y.value_counts())

from sklearn.model_selection import train_test_split #to perform the splitting
#train-test-split
test_pecentage = 0.3
data_rows = int(np.ceil(test_pecentage*len(data))) # amount of rows we gave for testing purposes
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_pecentage,random_state=42)#we will tale 30% of the dataset to testing and the rest for training
#by write random_state=42 we will get same data (same values) in train and test datasets every rime we ran this code.

first_train=x_train
first_test=x_test
check_train=y_train
check_test=y_test

# from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
log_reg = LogisticRegression(max_iter=2500) #max_iter parameter in the Logistic Regression model. This parameter determines the maximum number of iterations for the solver to converge.

# fit the model with data
log_reg.fit(first_train,check_train)

#make predictions on the testing set
predictionLog = log_reg.predict(first_test)


##Results##

#compute confusion_matrix by using Scikit-Learn’s confusion_matrix() function:
from sklearn.metrics import confusion_matrix
CMLog=confusion_matrix(check_test,predictionLog)
print(CMLog)#It's a 3x3 matrix because there are 3 response classes (0,1,2)


#we'll also check the accuracy by accuracy_score 
from sklearn import metrics
accurlog=metrics.accuracy_score(check_test, predictionLog)
print("The Accuracy of this model is:",accurlog) # accuracy 
print("the Classification Error for this model is:",1 - accurlog) #train error

#compute the loss using MEAN SQUARE ERROR
y_true = data.iloc[:data_rows].HeartDisease # took the first 1097 rows to fit the y_pred size, this is according to the data-train-test split.
y_pred = predictionLog #the acutal prediction 
loss = np.mean((np.array(y_true) - np.array(y_pred))**2) #MSE, in tensorflow -> tf.reduce_mean(tf.pow(pred - y_, 2))
print("The loss of this model using MSE calcualtion is:", loss)

#compute Cross-Entropy loss function for binary classification.
from sklearn.metrics import log_loss
epsilon = 1e-15  # Small constant to avoid log(0)
y_pred_entropy = predictionLog #the acutal prediction 
y_pred_entropy = np.clip(y_pred_entropy, epsilon, 1 - epsilon)  # Clip to avoid log(0)
ce_loss = -np.mean(y_true * np.log(y_pred_entropy) + (1 - y_true) * np.log(1 - y_pred_entropy))
ce_loss = log_loss(y_true, y_pred_entropy)
print("The loss of this model using Cross Entropy calcualtion is:", ce_loss)

# F1 Score:
tp = CMLog[0][0] #true positive
fp = CMLog[1][0] #false positive
fn = CMLog[0][1] #false negative
tn = CMLog[1][1] #true negative

recall = tp / (tp + fn) # True Positive / (True Positive + False Negative)
precision = tp / (tp + fp) # True Positive / (True Positive + False Positive)
f1 = 2*(precision * recall) / (precision + recall) #2 (Precision * Recall) / (Precision + Recall)
print("The F1-Measure for this model is :",f1)


#show CM as heat map
plt.figure(figsize=(10,10))#set size
sns.heatmap(CMLog, cmap="BuPu", annot=True, fmt='.0f', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title ('Logistic Regression Confusion Matrix')
#plt.show()

