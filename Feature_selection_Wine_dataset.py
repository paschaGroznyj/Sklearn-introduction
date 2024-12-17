import numpy as np
import pandas as pd

df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)
df_wine.columns = ['Метка класса', 'Алкоголь',
                   'Яблочная кислота', 'Зола',
                   'Щелочность золы', 'Магний',
                   'Всего фенолов', 'Флавоноиды',
                   'Нефлановидные фенолы', 'Проантоцианиды',
                   'Интенсивность цвета', 'Оттенок',
                   'OD280/OD315 разбавленных вин', 'Пролин']
print('Метки классов', np.unique(df_wine['Метка класса']))

from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
#НОРМАЛИЗУЕМ ДАННЫЕ
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_mms = mms.transform(X_test)
#СТАНДАРТИЗИРУЕМ ДАННЫЕ
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_train_std = std.fit_transform(X_train)
X_test_std = std.transform(X_test)
#ЧЕРЕЗ NUMPY
ex = np.array([0, 1, 2, 3, 4, 5])
ex_std = (ex - np.mean(ex))/np.std(ex)
ex_mms = (ex - ex.min())/(ex.max() - ex.min())

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l1', C= 2, solver='liblinear', multi_class='ovr')
lr.fit(X_train_std, y_train)
print('Правильность обучения ', lr.score(X_train_std, y_train))
print('Правильность обучения ', lr.score(X_test_std, y_test))
print(lr.intercept_)
print(lr.coef_[1])

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,8))
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan',
          'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.**c, solver='liblinear', multi_class='ovr', random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)
print(weights.shape[1]) #1 выведет 13 признаков, т.к массив 10х13
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], label = df_wine.columns[column+1],
             color = color)
plt.axhline(0, color = 'black', linestyle = '--', linewidth = 2)
plt.xlim([10**(-5), 10**5])
plt.ylabel('Весовой коэффициент')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc = 'upper left')
ax.legend(loc = 'upper center', bbox_to_anchor = (1.38, 1.03), ncol =1, fancybox = True)
plt.tight_layout()
plt.show()

from SBS_algorithm import SBS
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker = 'o')
plt.ylim([0.7, 1.02])
plt.ylabel('Точность')
plt.xlabel('Количество признаков')
plt.grid()
plt.tight_layout()
plt.show()

k3 = list(sbs.subsets_[10])
k = df_wine.columns[1:][k3].values#достаем столбцы начиная со второго, обращаемся по индексам k3

knn.fit(X_train_std, y_train)
print(knn.score(X_test_std, y_test))

knn.fit(X_train_std[:, k3], y_train)
print(knn.score(X_test_std[:, k3], y_test))

from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
print(importances)
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))
"""f + 1
Целое число для %2d. Оно отформатируется как " 1", " 2" и так далее.

30
Ширина для %s. Динамическое значение, задающее ширину столбца строки.

feat_labels[indices[f]]
Строка из списка feat_labels, которая будет отформатирована в поле шириной 30 символов.

importances[indices[f]]
Число с плавающей точкой для %f."""
plt.title('Важность признаков')
plt.bar(range(X_train.shape[1]), importances[indices])
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation = 90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f+1, 30, feat_labels[indices[f]], importances[indices[f]]))