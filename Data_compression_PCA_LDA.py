import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)

from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3, stratify=y)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# cov_mat = np.cov(X_train_std.T)
# eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
# print(eigen_vals, eigen_vecs)
#
# tot = sum(eigen_vals)
# var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
# cum_var_exp = np.cumsum(var_exp)
# print(cum_var_exp)
# import matplotlib.pyplot as plt
# plt.bar(range(1, 14), var_exp, alpha = 0.5, align='center', label = 'индивидуальная объясненная дисперсия')
# plt.step(range(1,14), cum_var_exp, where='mid', label = 'кумулятивная объясненная дисперсия')
# plt.ylabel('Коэффициент объясненной дисперсии')
# plt.xlabel('Индекс главного компонента')
# plt.legend(loc = "best")
# plt.tight_layout()
# plt.show()
# #создание пары собстенное значение-собственный вектор
# eigen_pair = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
# #сортируем кортежи(собственное значение, собственный вектор) от высоких к низким
# eigen_pair.sort(key=lambda k: k[0], reverse=True)
#
# w = np.hstack((eigen_pair[0][1][:, np.newaxis], eigen_pair[1][1][:, np.newaxis]))#получаем матрицу W
# """a = np.array([1, 2, 3])
# print(a.shape)  # (3,)
# b = a[:, np.newaxis]  # Добавляем новую ось
# print(b.shape)  # (3, 1)
# print(b)
# # [[1]
# #  [2]
# #  [3]]
# c = a[np.newaxis, :]  # Добавляем ось сверху
# print(c.shape)  # (1, 3)
# print(c)
# # [[1 2 3]]
#
# """
#
# print(X_train_std[0].dot(w))
# X_train_pca = X_train_std.dot(w)
# colors = ['r', 'b', 'g']
# markers = ['s', 'x', 'o']
# """y_train = np.array([0, 1, 0, 2, 1, 0])
# l = 1
# y_train == l
# # Вывод: array([False, True, False, False, True, False])
# X_train_pca = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
# y_train == l  # Булев массив: [False, True, False, False, True, False]
# X_train_pca[y_train == l]
# # Вывод: array([[3, 4], [9, 10]])
#
# """
# for l, c, m in zip(np.unique(y_train), colors, markers):
#     plt.scatter(X_train_pca[y_train==l, 0], X_train_pca[y_train==l, 1], c = c, label = l, marker = m)
# plt.xlabel("PC 1")
# plt.ylabel("PC 2")
# plt.legend(loc = 'lower left')
# plt.tight_layout()
# plt.show()
#
# from plot_decision_region_области_красок import plot_decision_regions
# from sklearn.linear_model import LogisticRegression
# from sklearn.decomposition import PCA
#
# pca = PCA(n_components=2)
# lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
# X_train_pca = pca.fit_transform(X_train_std)
# X_test_pca = pca.transform(X_test_std)
# print(X_train_pca)
# print(y_train)
# lr.fit(X_train_pca, y_train)
# plot_decision_regions(X_train_pca, y_train, classifier=lr)
# plt.xlabel("PC 1")
# plt.ylabel("PC 2")
# plt.legend(loc = 'lower left')
# plt.tight_layout()
# plt.show()
#
# plot_decision_regions(X_test_pca, y_test, classifier=lr)
# plt.xlabel("PC 1")
# plt.ylabel("PC 2")
# plt.legend(loc = 'lower left')
# plt.tight_layout()
# plt.show()

# np.set_printoptions(precision=4)
# mean_vecs = []
# for label in range(1, 4):
#     mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
#     print(mean_vecs[label-1])
# #Построение матрицы рассеяния внутри классов
# d = 13
# S_W = np.zeros((d, d))
# for label, mv in zip(range(1, 4), mean_vecs):
#     class_scatter = np.zeros((d, d))
#     for row in X_train_std[y_train == label]:
#         row, mv = row.reshape(d, 1), mv.reshape(d, 1)
#         class_scatter += (row - mv).dot((row - mv).T)
#     S_W += class_scatter
# #Получение масштабированной матрицы рассеяния внутри классов
# d = 13
# S_W = np.zeros((d, d))
# for label, mv in zip(range(1, 4), mean_vecs):
#     class_scatter = np.cov(X_train_std[y_train == label].T)
#     S_W += class_scatter
# #Рассчет матрицы Sb
# mean_overvall = np.mean(X_train_std, axis=0)
# d = 13
# S_B = np.zeros((d, d))
# for i, mean_vec in enumerate(mean_vecs):
#     n = X_train_std[y_train == i + 1].shape[0]
#     mean_vec = mean_vec.reshape(d, 1)
#     mean_overvall = mean_overvall.reshape(d, 1)
#     S_B += n*(mean_vec-mean_overvall).dot((mean_vec - mean_overvall).T)
#
# #Получение собственных значений и векторов
# import matplotlib.pyplot as plt
# eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
# eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
# eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
#
# tot = sum(eigen_vals.real)
# discr = [(i/tot) for i in sorted(eigen_vals.real, reverse=True)]
# cum_discr = np.cumsum(discr)
# plt.bar(range(1, 14), discr, alpha = 0.5, align='center', label = 'индивидуальная различимость')
# plt.step(range(1, 14), cum_discr, where='mid', label = 'кумулятивная различимость')
# plt.xlabel('Коэффициенты различимости')
# plt.ylabel('Линейные дискриминанты')
# plt.ylim([-0.1, 1.1])
# plt.legend(loc = 'best')
# plt.tight_layout()
# plt.show()
# w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))
#
# X_train_lda = X_train_std.dot(w)
# colors = ['r', 'b', 'g']
# markers = ['s', 'x', 'o']
#
# for l, c, m in zip(np.unique(y_train), colors, markers):
#     plt.scatter(X_train_lda[y_train == l, 0], X_train_lda[y_train == l, 1]*(-1), c = c, marker=m, label = l)
# plt.xlabel('LD 1')
# plt.ylabel('LD 2')
# plt.legend(loc = 'lower right')
# plt.tight_layout()
# plt.show()
#
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# lda = LinearDiscriminantAnalysis(n_components=2)
# X_train_lda = lda.fit_transform(X_train_std, y_train)
# X_test_lda = lda.transform(X_test_std)
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
# lr = lr.fit(X_train_lda, y_train)
# from plot_decision_region_области_красок import plot_decision_regions
# plot_decision_regions(X_test_lda, y_test, classifier=lr)
# plt.xlabel('LD 1')
# plt.ylabel('LD 2')
# plt.legend(loc = 'upper left')
# plt.tight_layout()
# plt.show()

#РЕАЛИЗАЦИЯ ЯДЕРНОГО АНАЛИЗА КОМПОНЕНТОВ
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
    """Параметры
    X: {np ndarray} [n_examples, n_features]
    gamma: float Параметр настройки ядра RBF
    Возвращает спроецированный набор данных
    X_pc: {np ndarray} [n_examples, n_features]"""
    #Вычислить попарные эвклидовы расстояния
    sq_dists = pdist(X, 'sqeuclidean')
    #Преобразовать попарные расстояния в квадратную матрицу
    mat_sq_dists = squareform(sq_dists)
    #вычислить симметричную матрицу ядра
    K = np.exp(-gamma*mat_sq_dists)
    #Центрировать матрицу ядра
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    #Получаем собственные пары из центрированной матрицы ядра
    eighvals, eigkvecs = eigh(K)
    eighvals, eigkvecs = eighvals[::-1], eigkvecs[:, ::-1]
    #Собрать верхние k собственных образцов
    X_pc = np.column_stack([eigkvecs[:, i] for i in range(n_components)])
    return X_pc
#
# #СОЗДАЕМ ЛУНУ
#
# from sklearn.datasets import make_moons
# X, y = make_moons(n_samples= 100, random_state=123)
# plt.scatter(X[y == 0, 0], X[y==0, 1], color = 'red', marker = '^', alpha=0.5)
# plt.scatter(X[y==1, 0], X[y==1, 1], color = 'blue', marker = 'o', alpha = 0.5)
# plt.tight_layout()
# plt.show()
#
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
#
#
# skicit_pca = PCA(n_components=2)  # Инициализация PCA с двумя компонентами.
# X_spca = skicit_pca.fit_transform(X)  # Преобразование данных X в пространство PCA (2 компоненты).
#
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))  # Создание двух субплотов.
#
# # Левый график
# ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1], color='red', marker='^', alpha=0.5, label='Class 0')
# ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1], color='blue', marker='o', alpha=0.5, label='Class 1')
# ax[0].set_xlabel('PC1')
# ax[0].set_ylabel('PC2')
# ax[0].legend()
#
# # Правый график
# ax[1].scatter(X_spca[y == 0, 0], np.zeros(50) + 0.02, color='red', marker='^', alpha=0.5)
# ax[1].scatter(X_spca[y == 1, 0], np.zeros(50) + 0.02, color='blue', marker='o', alpha=0.5)
# ax[1].set_xlabel('PC1')
# ax[1].set_ylim([-1, 1])  # Установка лимитов оси Y
#
# plt.tight_layout()  # Оптимизация расстояния между графиками
# plt.show()
#
# #ИСПЫТАНИЯ ФУНКЦИИ ЯДЕРНОГО PCA
# X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))  # Создание двух субплотов.
#
# # Левый график
# ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color='red', marker='^', alpha=0.5, label='Class 0')
# ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color='blue', marker='o', alpha=0.5, label='Class 1')
# ax[0].set_xlabel('PC1')
# ax[0].set_ylabel('PC2')
# ax[0].legend()
#
# # Правый график
# ax[1].scatter(X_kpca[y == 0, 0], np.zeros(50) + 0.02, color='red', marker='^', alpha=0.5)
# ax[1].scatter(X_kpca[y == 1, 0], np.zeros(50) + 0.02, color='blue', marker='o', alpha=0.5)
# ax[1].set_xlabel('PC1')
# ax[1].set_ylim([-1, 1])  # Установка лимитов оси Y
#
# plt.tight_layout()  # Оптимизация расстояния между графиками
# plt.show()

from sklearn.datasets import make_circles

X, y = make_circles(n_samples=1000, random_state= 123, noise=0.1, factor=0.2)
plt.scatter(X[y == 0, 0], X[y==0, 1], color = 'red', marker = '^', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color = 'blue', marker = 'o', alpha=0.5)
plt.tight_layout()
plt.show()

from sklearn.decomposition import PCA

scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))  # Создание двух субплотов.

# Левый график
ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1], color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')


# Правый график
ax[1].scatter(X_spca[y == 0, 0], np.zeros(500) + 0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y == 1, 0], np.zeros(500) - 0.02, color='blue', marker='o', alpha=0.5)
ax[1].set_xlabel('PC1')
ax[1].set_ylim([-1, 1])  # Установка лимитов оси Y

plt.tight_layout()  # Оптимизация расстояния между графиками
plt.show()

X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))  # Создание двух субплотов.

# Левый график
ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')


# Правый график
ax[1].scatter(X_kpca[y == 0, 0], np.zeros(500) + 0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y == 1, 0], np.zeros(500) - 0.02, color='blue', marker='o', alpha=0.5)
ax[1].set_xlabel('PC1')
ax[1].set_ylim([-1, 1])  # Установка лимитов оси Y

plt.tight_layout()  # Оптимизация расстояния между графиками
plt.show()

#МОДЕРНИЗАЦИЯ ЯДЕРНОГО АНАЛИЗА КОМПОНЕНТОВ
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
    """Параметры
    X: {np ndarray} [n_examples, n_features]
    gamma: float Параметр настройки ядра RBF
    Возвращает спроецированный набор данных
    X_pc: {np ndarray} [n_examples, n_features]"""
    #Вычислить попарные эвклидовы расстояния
    sq_dists = pdist(X, 'sqeuclidean')
    #Преобразовать попарные расстояния в квадратную матрицу
    mat_sq_dists = squareform(sq_dists)
    #вычислить симметричную матрицу ядра
    K = np.exp(-gamma*mat_sq_dists)
    #Центрировать матрицу ядра
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    #Получаем собственные пары из центрированной матрицы ядра
    eighvals, eigkvecs = eigh(K)
    eighvals, eigkvecs = eighvals[::-1], eigkvecs[:, ::-1]
    #Собрать верхние k собственных образцов
    alphas = np.column_stack([eigkvecs[:, i] for i in range(n_components)])
    lambdas = [eighvals[i] for i in range(n_components)]
    return alphas, lambdas

from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, random_state=123)
print(X)
print(y)
alphas, lambdas = rbf_kernel_pca(X, 15, 1)
x_new = X[25]
print(x_new)
x_proj = alphas[25]
print(x_proj)
def projest_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum(x_new-row)**2 for row in X])
    k = np.exp(-gamma*pair_dist)
    return k.dot(alphas/lambdas)
x_reproj = projest_x(x_new, X, 15, alphas, lambdas)
print(x_reproj)

plt.scatter(alphas[y==0, 0], np.zeros((50)), color = 'red', marker = '^', alpha=0.5)
plt.scatter(alphas[y==1, 0], np.zeros((50)), color = 'blue', marker = 'o', alpha=0.5)
plt.scatter(x_proj, 0, color= 'black', label = 'Первоначальная проекция точки Х[25]', marker = '^', s = 100)
plt.scatter(x_reproj, 0, color= 'black', label = 'Повторно отображенная точка Х[25]', marker = 'x', s = 100)
plt.legend(scatterpoints = 1)
plt.tight_layout()
plt.show()

from sklearn.decomposition import KernelPCA

X, y = make_moons(n_samples=100, random_state=123)
scikit_kpca = KernelPCA(kernel='rbf', n_components=2, gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)
plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1], color = 'red', marker='x', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1], color = 'blue', marker = '^', alpha=0.5)
plt.xlabel("PC 1")
plt.ylabel('PC 2')
plt.tight_layout()
plt.show()
