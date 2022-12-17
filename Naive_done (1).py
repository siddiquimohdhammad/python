'''
Adnan Sulaiman
20co13
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = np.array(
    [[0,1232,"Male",19,19000,0],
                    [1,2123,"Male",25,45000,0],
                    [2,2342,"Male",25,43000,1],
                    [3,2342,"Male",25,11000,1],
                    [4,2312,"Female",20,46000,1],
                    [5,423,"Male",27,58000,0],
                   [6,3456,"Female",27,84000,0],
                   [7,55,"Female",32,150000,1],
                   [8,123,"Male",25,33000,0],
                   [9,8678,"Female",35,65000,0]]
    )

X = dataset[:, [2, 3,4]]
print("x is ........", X)
y = dataset[:, -1]
print("y is ........", y)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,0] = le.fit_transform(X[:,0])
print('le is ...',X)



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print("Accuracy:-",ac)


'''
Accuracy:- 0.5
Confusion matrix:- 
[[2 2]
 [0 0]]
'''