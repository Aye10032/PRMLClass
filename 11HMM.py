import numpy as np

from fractions import Fraction

np.set_printoptions(formatter={'all': lambda x: str(Fraction(x).limit_denominator())})

states = ['box1', 'box2', 'box3']
observables = ['a', 'o']

pi = np.array([1 / 3, 1 / 3, 1 / 3])
A = np.array([[1 / 3, 1 / 3, 1 / 3],
              [1 / 3, 1 / 3, 1 / 3],
              [1 / 3, 1 / 3, 1 / 3]])
B = np.array([[1 / 2, 1 / 2],
              [3 / 4, 1 / 4],
              [1 / 4, 3 / 4]])
obs_sequence = ['a', 'a', 'o', 'o', 'o']
obs_idx = [observables.index(x) for x in obs_sequence]


def viterbi(_obs_sequence, _pi, _A, _B):
    T = len(_obs_sequence)
    N = len(states)
    V = np.zeros((T, N))
    path = np.zeros((T, N), dtype=int)
    for t in range(T):
        if t == 0:
            V[t] = _pi * _B[:, obs_idx[t]]

            print('----------init-----------')
            print(V[0])
        else:
            print(f'--------iter {t}----------')
            for j in range(N):
                V[t, j] = np.max(V[t - 1] * A[:, j]) * B[j, obs_idx[t]]
                path[t, j] = np.argmax(V[t - 1] * A[:, j])
            print(V[t])

    print('----------end-----------')
    print(V.T)
    print('------------------------')
    print(path.T)

    print('----------回溯-----------')
    y = np.zeros(T, dtype=int)
    y[-1] = np.argmax(V[-1])

    for t in range(T - 1, 0, -1):
        y[t - 1] = path[t, y[t]]

    print(y)


viterbi(obs_sequence, pi, A, B)
