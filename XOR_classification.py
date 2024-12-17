import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
np.random.seed(1)#задаем стартовое значение для генератора случ чисел
X_xor = np.random.randn(200, 2)#генерирует случайные числа нормального распределения от -1 до 1, 2 столбца, 200 строк
print(X_xor)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
print(y_xor)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(x = X_xor[y_xor == 1, 0], y = X_xor[y_xor == 1, 1], c = 'b', marker= 'x', label = '1')
plt.scatter(x = X_xor[y_xor == -1, 0], y = X_xor[y_xor == -1, 1, ], c = 'r', marker= 's', label = '-1')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()

def plot_decision_regions(X, y, classifier, test_idx = None, resolution = 0.2):
    #настраиваем цветную карту
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    #выводим поверность решения
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    print(Z)
    Z = Z.reshape(xx1.shape)
    print(Z)
    #заливаем получившиеся контура цветами
    plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    """
    names = ["John", "Jane", "Doe"]
    enumNames = enumerate(names, 10)

    print(list(enumNames))
    # [(10, 'John'), (11, 'Jane'), (12, 'Doe')]
    """
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y = X[y==cl, 1], alpha=0.8, c = colors[idx],
                    marker=markers[idx], label = cl,
                    edgecolors='black')
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(x=X_test[:, 0], y = X_test[:, 1], c = colors[0], edgecolors='black', alpha=0.9,
                    linewidths=1, marker='o', s = 30, label = 'испытательный набор')

from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=1, gamma=0.4, C = 100.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc = 'upper left')
plt.show()
print('Правильность: %.3f' % svm.score(X_xor, y_xor))
