from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from keras.datasets import cifar10
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from loguru import logger

logger.add('log/svm.log')

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255, x_test / 255

X = x_train.reshape(50000, 3 * 32 * 32)
Xt = x_test.reshape(10000, 3 * 32 * 32)
y = y_train.flatten()
yt = y_test.flatten()

logger.info('Start PCA')

pca = PCA(n_components=0.95)
X = pca.fit_transform(X)
Xt = pca.transform(Xt)

logger.info('Start choose parameters')

param_grid = dict(
    gamma=[1e-3, 1e-2, 0.1, 1, 10, 100],
    C=[1, 1.5, 2, 3, 5],
    kernel=['rbf', 'linear', 'sigmoid']
)
model = SVC(decision_function_shape='ovo', verbose=1)

grid_search = GridSearchCV(model, param_grid)
grid_search.fit(X[:10000], y[:10000])

C = grid_search.best_params_['C']
gamma = grid_search.best_params_['gamma']
kernel = grid_search.best_params_['kernel']
logger.info(f'Best Parameters: C:{C}, gamma:{gamma}, kernel:{kernel}')
logger.info(f'Best Score: {grid_search.best_score_}')

logger.info(f'start training the model with C={C}, gamma={gamma}, kernel={kernel}')
clf = SVC(C=C, gamma=gamma, kernel=kernel, decision_function_shape='ovo', verbose=11)
clf.fit(X, y)
logger.info('training finished')

y_pred_train = clf.predict(X)
y_pred_test = clf.predict(Xt)

train_accuracy = accuracy_score(y, y_pred_train)
test_accuracy = accuracy_score(yt, y_pred_test)

logger.info(f'训练集准确度: {train_accuracy}')
logger.info(f'测试集准确度: {test_accuracy}')
