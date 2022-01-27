import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from matplotlib import cm
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


transfusion_data = pd.read_csv('transfusion.csv', sep=',')

# Renaming the column titles and shortening the names of them
transfusion_data.columns = ["Recency", "Frequency", "Monetary", "Time", "Donated_2007"]
print("Total number of Attributes: 4")
print("Index: Donated in 2007")
print("Distance Metric: Eucledian")
print("Training Size: 75%")
print("Testing Size: 25%")

X = transfusion_data[['Recency', 'Frequency', 'Monetary', 'Time']]
y = transfusion_data['Donated_2007']

# uses default Eucledian distance
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# random_state: set seed for random# generator
# test_size: default 25% testing, 75% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

# Train the classifier (fit the estimator) using the training data
knn.fit(X_train, y_train)

# Create confusion matrix
print ("Confusion Matrix:")
y_pred = knn.predict(X_test)
cf_matrix = confusion_matrix(y_test, y_pred)

print(cf_matrix)

# Estimate the accuracy of the classifier on future data, using the test data
accuracy = knn.score(X_test, y_test)

# How sensitive is k-NN classification accuracy to the choice of the 'k' parameter?
k_range = range(1, 20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

# visualization
# plotting a scatter matrix
cmap = cm.get_cmap('gnuplot')
scatter = scatter_matrix(X, c=y, marker='o', s=40, hist_kwds={'bins': 15}, figsize=(9, 9), cmap=cmap)
scatter.view()

# plotting a 3D scatter plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d   # must keep
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X['Recency'], X['Frequency'], X['Monetary'], c=y, marker='o', s=100)
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')

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


