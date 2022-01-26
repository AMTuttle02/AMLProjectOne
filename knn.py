
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# predict the class of instance 1.1
print(knn.predict([[1.1]]))
# with probabilities
print(knn.predict_proba([[1.1]]))

import pandas as pd
bmi = pd.read_csv('bmi.csv')

gender = {'Male': 1,'Female': 2}

bmi.Gender = [gender[item] for item in bmi.Gender]

bmi.head()

X = bmi[['Gender', 'Height', 'Weight']]
y = bmi['Index']

knn.fit(X, y)

unknown1 = pd.DataFrame([[1, 174, 96]], columns=['Gender', 'Height', 'Weight'])
bmi_prediction = knn.predict(unknown1)

# Describes what each index value is based on the array location
bmiDescription = ['Extremely Weak', 'Weak', 'Normal', 'Overweight', 'Obesity', 'Extreme Obesity']

# print user friendly meaning of Index value
print(bmiDescription[bmi_prediction[0]])

# print probability of each index value
print(knn.predict_proba(unknown1))

# second example:
unknown2 = pd.DataFrame([[1, 190, 55]], columns=['Gender', 'Height', 'Weight'])
bmi_prediction = knn.predict(unknown2)

# print user friendly meaning of Index value
print(bmiDescription[bmi_prediction[0]])

# print probability of each index value
print(knn.predict_proba(unknown1))

from sklearn.model_selection import train_test_split
#random_state: set seed for random# generator
#test_size: default 25% testing, 75% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

# Train the classifier (fit the estimator) using the training data
knn.fit(X_train, y_train)

# Estimate the accuracy of the classifier on future data, using the test data
knn.score(X_test, y_test)

# How sensitive is k-NN classification accuracy to the choice of the 'k' parameter?
k_range = range(1, 20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

# visualization
# plotting a scatter matrix
from matplotlib import cm
from pandas.plotting import scatter_matrix
cmap = cm.get_cmap('gnuplot')
scatter = scatter_matrix(X, c=y, marker='o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)

# plotting a 3D scatter plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d   # must keep
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X['Gender'], X['Height'], X['Weight'], c = y, marker = 'o', s=100)
ax.set_xlabel('Gender')
ax.set_ylabel('Height')
ax.set_zlabel('Weight')

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0, 5, 10, 15, 20])

# How sensitive is k-NN classification accuracy to the train/test split proportion?
import numpy as np
t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
knn = KNeighborsClassifier(n_neighbors=5)
plt.figure()
for s in t:
    scores = []
    for i in range(1, 1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')
plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy')
plt.show()
