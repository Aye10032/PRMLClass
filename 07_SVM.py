import numpy as np
from sklearn.svm import SVC

X = np.array([[3, 3], [4, 3], [1, 1]])
y = np.array([1, 1, -1])

svm = SVC(kernel='linear')
svm.fit(X, y)

w = svm.coef_[0]
b = svm.intercept_[0]

print(w)
print(b)
