import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
X, y = datasets.make_classification(
    n_samples=500, 
    n_features=2, 
    n_redundant=0, 
    n_informative=2, 
    random_state=42, 
    n_clusters_per_class=1
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
new_samples = [[0.5, -1.2], [-1.5, 1.8]]
predictions = svm.predict(new_samples)
print("\nPredictions for new samples:", predictions)