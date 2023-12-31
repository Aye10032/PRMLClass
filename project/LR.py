from keras.datasets import cifar10
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logger.add('log/lr.log')

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

logger.info(f'start training the model')
lr = LogisticRegression(solver='sag', random_state=4, verbose=1)
lr.fit(X, y)
logger.info('training finished')

y_pred_train = lr.predict(X)
y_pred_test = lr.predict(Xt)

train_accuracy = accuracy_score(y, y_pred_train)
test_accuracy = accuracy_score(yt, y_pred_test)

logger.info(f'训练集准确度: {train_accuracy}')
logger.info(f'测试集准确度: {test_accuracy}')
