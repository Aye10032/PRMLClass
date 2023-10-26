import numpy as np


def perceptron_train(_patterns, _labels, learning_rate, epochs):
    _patterns = np.hstack((patterns, np.ones((_patterns.shape[0], 1))))
    print(f'{_patterns}\n')
    _w = np.zeros((_patterns.shape[1], _patterns.shape[0]))
    print(f'init w=\n{_w}\n')

    for epoch in range(epochs):
        print(f'epoch{epoch}:')
        w_update = False
        for i, pattern in enumerate(_patterns):
            _d = np.dot(_w, np.transpose(pattern))
            max_label = np.where(_d == np.max(_d))[0]

            print(f'  for x{i + 1}={pattern}, now w={_w.flatten()}, max_label={max_label}, \n', end='')

            if max_label.__len__() == 1 and max_label[0] == i:
                print(f'    keep w\n')
            else:
                _w += learning_rate * np.outer(labels[i], pattern)
                print(f'    update w={_w.flatten()}\n')
                w_update = True

        if not w_update:
            break

    return _w


patterns = np.array([[-1, -1],
                     [0, 0],
                     [1, 1]])
labels = np.array([[1, -1, -1],
                   [-1, 1, -1],
                   [-1, -1, 1]])

w = perceptron_train(patterns, labels, 1, 10)

print(f'w = {w}')
