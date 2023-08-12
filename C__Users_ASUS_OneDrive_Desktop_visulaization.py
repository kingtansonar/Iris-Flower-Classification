#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[21]:


import pandas as pd


iris_data = pd.read_csv("iris.csv")


print(iris_data.head())


X = iris_data.drop("SepalLengthCm", axis=1)  
y = iris_data["PetalLengthCm"]


# In[23]:


X = iris_data.drop('Species', axis=1)  
y = iris_data['Species'] 


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)


knn.fit(X_train, y_train)


# In[25]:


y_pred = knn.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[29]:


iris_data = pd.read_csv("iris.csv")
X = iris_data.drop("SepalLengthCm", axis=1)
y = iris_data["PetalLengthCm"] 


# In[31]:


from sklearn.tree import DecisionTreeClassifier


classifier = DecisionTreeClassifier(random_state=42)


classifier.fit(X_train, y_train)


# In[34]:


y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[36]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(iris_data, hue="Species")
plt.show()


# In[38]:


sns.boxplot(x="Species", y="PetalLengthCm", data=iris_data)
plt.show()


# In[40]:


sns.violinplot(x="Species", y="PetalLengthCm", data=iris_data)
plt.show()


# In[42]:


corr_matrix = iris_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()


# In[50]:





# In[51]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap


iris = load_iris()
X = iris.data[:, :2] 
y = iris.target


classifier = DecisionTreeClassifier()
classifier.fit(X, y)


h = 0.02 
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


cmap_background = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_points = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_points, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary Visualization')
plt.show()

