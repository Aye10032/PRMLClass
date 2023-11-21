from sklearn.naive_bayes import GaussianNB
import numpy as np

label = {'S': 0, 'M': 1, 'L': 2, 'N': 3}

x1 = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
x2 = ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']
y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]

X = np.array([x1, [label.get(x, -1) for x in x2]]).T

clf = GaussianNB()

clf.fit(X, y)

test_samples = [[2, label['S']], [1, label['N']]]
predictions = clf.predict(test_samples)

for index, predict in enumerate(predictions):
    print(f'sample{index}: {predict}')
