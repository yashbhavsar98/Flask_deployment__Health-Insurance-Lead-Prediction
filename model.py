import numpy as np
import pandas as pd
import pickle

train = pd.read_csv('train.csv')

train = train.drop(['ID','City_Code','Region_Code'], axis =1)

# Data Cleaning

#Changing names for better prediction.
train.Accomodation_Type[train.Accomodation_Type == 'Rented'] = 1
train.Accomodation_Type[train.Accomodation_Type == 'Owned'] = 0

train.Reco_Insurance_Type[train.Reco_Insurance_Type == 'Individual'] = 0
train.Reco_Insurance_Type[train.Reco_Insurance_Type == 'Joint'] = 1

train.Is_Spouse[train.Is_Spouse == 'Yes'] = 1
train.Is_Spouse[train.Is_Spouse == 'No'] = 0

train.Health_Indicator[train.Health_Indicator == 'X1'] = 1
train.Health_Indicator[train.Health_Indicator == 'X2'] = 2
train.Health_Indicator[train.Health_Indicator == 'X3'] = 3
train.Health_Indicator[train.Health_Indicator == 'X4'] = 4
train.Health_Indicator[train.Health_Indicator == 'X5'] = 5
train.Health_Indicator[train.Health_Indicator == 'X6'] = 6
train.Health_Indicator[train.Health_Indicator == 'X7'] = 7
train.Health_Indicator[train.Health_Indicator == 'X8'] = 8
train.Health_Indicator[train.Health_Indicator == 'X9'] = 9

#Converting datatype float to int of policy premium
train['Reco_Policy_Premium'] = train['Reco_Policy_Premium'].astype(int)

#Filling null values.
#Replacing health indicator nulls with mode
mode0 = train['Health_Indicator'].mode()[0]
train['Health_Indicator']= train['Health_Indicator'].fillna(mode0)

#Also converting 14+ with only 14 as to make all the values as ojbect variable.
#Replacing holding policy duration nulls with median as our dataset is right skewed
train['Holding_Policy_Duration'] = train['Holding_Policy_Duration'].replace('14+',14).astype(float)
train['Holding_Policy_Duration'] = train['Holding_Policy_Duration'].fillna(train['Holding_Policy_Duration'].median())

#Filling nulls of Holdinf_Policy_Type with median value
train['Holding_Policy_Type'] = train['Holding_Policy_Type'].fillna(train['Holding_Policy_Type'].median())

#Dropping the target variable and assigning it to variable y
X = train.drop('Response', axis = 1)
y = train['Response']

#Making a KNN model for prediction 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Data splitting into train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state = 42 )

#Model Training with KNN having neighbors as 6
knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
kn_score_test = knn.score(X_test,y_test)
kn_score_train= knn.score(X_train,y_train)
print("Accuracy: " +str(accuracy_score(y_test, y_pred)))
print("Train Score:" +str( kn_score_train))

#Dumping Data intp .pkl file
pickle.dump(knn,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

