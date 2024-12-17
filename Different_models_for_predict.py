from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score #для получения метрик эффективности распознования
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
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
        """
        y_test = [2 0 0 2 1 1 2 1 2 0 0 2 0 1 0 1 2 1 1 2 2 0 1 2 1 1 1 2 0 2 0 0 1 1 2 2 0
 0 0 1 2 2 1 0 0]
        X_test = [[ 0.89820289  1.44587881]
 [-1.16537974 -1.04507821]
 [-1.33269725 -1.17618121]
 [ 0.39625036  0.65926081]
 [ 0.34047786  0.2659518 ]
 [ 0.11738784  0.1348488 ]
        """
    """X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])  # Координаты точек
y = np.array([0, 1, 0, 2, 1, 2])  # Метки классов
cl = 1  # Текущий класс

# Булевый массив
mask = y == cl  # array([False, True, False, False, True, False])

# Индексация массива X
X[mask, :]      # array([[ 3,  4], [ 9, 10]]) — точки класса 1
X[mask, 0]      # array([3, 9]) — x-координаты
X[mask, 1]      # array([4, 10]) — y-координаты
    """
"""Загружаем данные из встроенной функции
"""
iris = datasets.load_iris()
X = iris.data[:, [2,3]]#Берем 3 и 4 столбцы
y = iris.target
print(np.unique(y))#0, 1, 2 Наименования цветов
#разбиваем на тестовую и обучающую выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
print((np.bincount(y))) #[50 50 50] всего 150 образцов
print((np.bincount(y_train)))#[35 35 35] 105 на обучение
print((np.bincount(y_test)))#[15 15 15] 45 на тесты
#Стандартизируем признаки используя класс StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
print(X_test_std[:3, :])
#Создаем персептрон с помощью класса Perseptron
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
print('Неправильно классифицированных образцов: %d'%(y_test != y_pred).sum())
#или с помощью metrics
print('Правильность: %.3f' % accuracy_score(y_pred, y_test))
#еще один метод оценки предсказаний
print('Правильность: %.3f' % ppn.score(X_test_std, y_test))

X_combined_std = np.vstack((X_train_std, X_test_std))#объединяем по вертикали, т.е возвращаем свою выборку X
y_combined = np.hstack((y_train, y_test))#объединяем по горизонтали, т.е возвращаем выборку y

plot_decision_regions(X_combined_std, y_combined, classifier=ppn, test_idx=range(105, 150))
plt.xlabel('длина лепестка')
plt.ylabel('ширина лепестка')
plt.legend(loc='upper left')
plt.title('Perseptron')
plt.tight_layout() #автоматическое расположение элментов, чтобы они не перекрывали друг друга
plt.show()

from sklearn.linear_model import LogisticRegression
#OVR или полиномиальная логическая регрессия
#solver параметры оптимизации
lr = LogisticRegression(C=10, random_state=1, solver='lbfgs', multi_class='ovr')
lr.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.title('LogisticRegression')
plt.tight_layout()

# plt.savefig('images/03_06.png', dpi=300)
plt.show()
print('Правильность: %.3f' % lr.score(X_test_std, y_test))
lr.predict_proba(X_test_std[:3, :])
"""array([[3.81527885e-09, 1.44792866e-01, 8.55207131e-01],
       [8.34020679e-01, 1.65979321e-01, 3.25737138e-13],
       [8.48831425e-01, 1.51168575e-01, 2.62277619e-14]])
       Вероятность принадлежности образцов к одному из классов"""
lr.predict_proba(X_test_std[:3, :]).argmax(axis=1)
#array([2, 0, 0])
lr.predict(X_test_std[:3, :])
#array([2, 0, 0]) более удобный метод
"""import numpy as np
работа с reshape, где 1 это требуемое количество строк, а -1 это автоматическое определение
количества столбцов
# Исходный массив
arr = np.array([1, 2, 3, 4])  # 1D массив с 4 элементами

# Преобразование в двумерный массив с 1 строкой
reshaped = arr.reshape(1, -1)
print(reshaped)
# [[1 2 3 4]] — двумерный массив: 1 строка, 4 столбца

print(reshaped.shape)
# (1, 4)
"""
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier=svm,
                      test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.title('SVC-linear')
plt.tight_layout()
#plt.savefig('images/03_11.png', dpi=300)
plt.show()
print('Правильность: %.3f' % svm.score(X_test_std, y_test))

svm = SVC(kernel='rbf', gamma=1, C = 1.0, random_state= 1)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, test_idx=range(105, 150), classifier=svm)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.title('SVC-rbf')
plt.tight_layout()
plt.show()
##########################На ОСНОВЕ ДЕРЕВА ПРИНЯТИЯ РЕШЕНИЙ
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
tree_model = DecisionTreeClassifier(criterion='gini',
                                    max_depth=10,
                                    random_state=1)
tree_model.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined,
                      classifier=tree_model,
                      test_idx=range(105, 150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.title('Tree')
plt.tight_layout()
#plt.savefig('images/03_20.png', dpi=300)
plt.show()
tree.plot_tree(tree_model)
plt.show()

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1)
forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
plt.xlabel('длина лепестка')
plt.ylabel('ширина лепестка')
plt.legend(loc = 'upper left')
plt.title('Forest')
plt.tight_layout()
plt.show()

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, p = 2, metric='minkowski')
knn.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(105,150))
plt.xlabel('длина лепестка[стандартизированная]')
plt.ylabel('ширина лепестка[стандартизированная]')
plt.legend(loc = 'upper left')
plt.title('KNN')
plt.tight_layout()
plt.show()
