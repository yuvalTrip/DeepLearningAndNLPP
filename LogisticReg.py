import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data_NaN = pd.read_csv(".\DeepLearningAndNLPP\heart-disease-data.csv")
data = data_NaN.dropna().reset_index(drop=True) # cleared NaN values and reset id's 
print(data)

#train = data.iloc[:2200] # first rows to train the dataset
#test = data.iloc[2200:3000] # other rows to test on the data set
#running_data = data.iloc[3000:] #rest of the data for checking if the model predicts correctly

y=data.TenYearCHD #strength is the label we want to predict
x=data.drop('TenYearCHD',axis=1)#use drop function to take all other data in x


from sklearn.model_selection import train_test_split #to perform the splitting
#train-test-split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)#we will tale 30% of the dataset to testing and the rest for training
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

#compute confusion_matrix by using Scikit-Learn’s confusion_matrix() function:
from sklearn.metrics import confusion_matrix
CMLog=confusion_matrix(check_test,predictionLog)
print(CMLog)#It's a 3x3 matrix because there are 3 response classes (0,1,2)


#we'll also check the accuracy by accuracy_score 
from sklearn import metrics
accurlog=metrics.accuracy_score(check_test, predictionLog)
print("The Accuracy of this model is:",accurlog)
print("the Classification Error for this model is:",1 - accurlog)
      
#show CM as heat map
plt.figure(figsize=(10,10))#set size
sns.heatmap(CMLog,cmap = "BuPu", annot=True, fmt = '.0f')
plt.title ('Logistic Regression Confusion Matrix')
#plt.show()

