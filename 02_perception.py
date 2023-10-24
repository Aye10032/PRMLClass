import numpy as np


def perceptron_train(_patterns, _labels, learning_rate, epochs):
    _patterns = np.hstack((patterns, np.ones((_patterns.shape[0], 1))))
    print(f'{_patterns}\n')
    _w = np.zeros(_patterns.shape[1])
    print(f'init w={_w}\n')

    for epoch in range(epochs):
        print(f'epoch{epoch}:')
        w_update = False
        for i, pattern in enumerate(_patterns):
            predicted_label = np.sign(np.dot(pattern, _w))

            print(f'  for x{i}={pattern},label={_labels[i]}, now w={_w}, predicted_label={predicted_label},', end='')
            if predicted_label != _labels[i]:
                _w += learning_rate * _labels[i] * pattern
                w_update = True
                print(f'update w={_w}')
            else:
                print(f'keep w')

        if not w_update:
            break

    return _w


patterns = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0],
                     [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1]])
labels = np.array([1, 1, 1, 1, -1, -1, -1, -1])

w = perceptron_train(patterns, labels, 1, 10)

print(f'w = {w}')
