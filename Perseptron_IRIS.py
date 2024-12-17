import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print('URL:', s)

df = pd.read_csv(s, header=None, encoding='utf-8')
y = df.iloc[0:100, 4].values #выбираем классы соответсвующие названию цветку, 4 столбец
y = np.where(y == 'Iris-setosa', -1, 1)#создаем массив с условием
#извлекаем длину чашелистика и длину лепестка, двумерный массив
X = df.iloc[0:100, [0,2]].values
print(X)
plt.scatter(x= X[0:50, 0], y = X[0:50, 1], color = 'red', marker='o', label = 'щетинистый')
plt.scatter(X[50:100, 0], X[50:100, 1], color = 'blue', marker='x', label = 'разноцветный')
plt.xlabel('длина чашелистика см')
plt.ylabel('длина лепестка см')
plt.legend(loc = 'upper left')
plt.show()

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    #создаем карту цветов
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1#находим мин и макс для первого признака
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    """
    X = np.array([
    [1, 2],  # Точка класса 1
    [2, 3],  # Точка класса 1
    [3, 1],  # Точка класса -1
    [4, 2],  # Точка класса -1
            ])
x1_min, x1_max = 0, 5
x2_min, x2_max = 0, 4
resolution = 1
np.arange(0, 5, 1) → [0, 1, 2, 3, 4]
np.arange(0, 4, 1) → [0, 1, 2, 3]
xx1 = [[0, 1, 2, 3, 4],
       [0, 1, 2, 3, 4],
       [0, 1, 2, 3, 4],
       [0, 1, 2, 3, 4]]

xx2 = [[0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1],
       [2, 2, 2, 2, 2],
       [3, 3, 3, 3, 3]]
xx1.ravel() → [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, ...]
xx2.ravel() → [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, ...]
np.array ->[[0, 0],
             [1, 0],
             [2, 0],
             [3, 0],
             [4, 0],
             [0, 1],
             [1, 1],
             ...] Затем этот массив передается в функцию predict для заполения 1 и -1
    """
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')


class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0) скорость обучения
    n_iter : int
      Passes over the training dataset. переходы по обучающему нобору данных
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting. Веса после подгонки
    errors_ : list
      Number of misclassifications (updates) in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.
    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples(количество образцов) and
      n_features is the number of features(количество признаков).
    y : array-like, shape = [n_examples]
      Target values(Целевые значения).
    Returns
    -------
    self : object
    """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        """
X = [[1.2, 0.7, -0.3],
     [0.4, -1.2, 2.3],
     [3.1, 0.1, -0.8]]
     n_examples = 3 (3 строки — количество объектов).
n_features = 3 (3 столбца — количество признаков на объект).
y = [1, 0, 1]
xi — это одна строка из X, то есть одномерный массив длиной n_features (например: [1.2, 0.7, -0.3]).
target — соответствующее значение из y (например: 1).
        """
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))#передаем массив xi  функцию, получаем 1 или 0
                self.w_[1:] += update * xi
                self.w_[0] += update
                """
                self.w_ = [0.5, 0.2, -0.4]  # Изначальные веса
                xi = [1.0, 2.0]             # Признаки текущего объекта
                update = 0.1                # Величина обновления
                
                self.w_[1:] += update * xi
                # update * xi = [0.1 * 1.0, 0.1 * 2.0] = [0.1, 0.2]
                # self.w_[1:] = [0.2, -0.4] + [0.1, 0.2] = [0.3, -0.2]
                """
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    #скалярное произведение матриц и добавляем биос

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    #возвращает 1 если больше 0 и -1 в противном случае

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker = 'o')
plt.xlabel('Эпохи')
plt.ylabel('Количество обновлений')
plt.show()

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

# plt.savefig('images/02_08.png', dpi=300)
plt.show()
