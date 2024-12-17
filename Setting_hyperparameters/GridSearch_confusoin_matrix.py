import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv("wdbc.data", header=None)
X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

pile_lr = make_pipeline(StandardScaler(),
                   PCA(n_components=2),
                   LogisticRegression(random_state=1, solver = 'lbfgs'))
pile_lr.fit(X_train, y_train)
y_pred = pile_lr.predict(X_test)
print('Правильность испытаний: %.3f' % pile_lr.score(X_test, y_test))

from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
scores = []
"""train - массив индексов строк обучающего набора, всего в kfold 10 таких массивов,
для test - тоже идут массивы индексов строк"""
for k, (train, test) in enumerate(kfold):#Извлекаем соответсвующие строки тренировочного и тестового набора
    pile_lr.fit(X_train[train], y_train[train])
    score = pile_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Блок: %2d, Распеределение классов: %s, Правильность: %.3f' % (k+1, np.bincount(y_train[train]), score))
print("Точность перекрестной проверки: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=pile_lr, X = X_train, y = y_train, cv = 10, n_jobs=1)
print(scores)
print("Точность перекрестной проверки: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))

import matplotlib.pyplot as plt
# from sklearn.model_selection import learning_curve
#
# pile_lr = make_pipeline(StandardScaler(),
#                         LogisticRegression(penalty='l2', random_state=1, solver='lbfgs', max_iter=10000))
# train_sizes, train_scores, test_scores = learning_curve(estimator=pile_lr, X = X_train, y = y_train,
#                                                         train_sizes=np.linspace(0.1, 1.0, 10),
#                                                         cv = 10, n_jobs= 1)#lispace разбивает диапазон на число участков num
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)
# plt.plot(train_sizes, train_mean, color = 'blue', marker = 'o', markersize = 5,
#          label = 'правильность при обучении')
# plt.fill_between(train_sizes, train_mean+train_std, train_mean-train_std, alpha = 0.15, color = 'blue')
# plt.plot(train_sizes, test_mean, color = 'green', linestyle = '--', marker = 's', markersize = 5,
#          label = 'правильность при проверке')
# plt.fill_between(train_sizes, test_mean+test_std, test_mean-test_std, color = 'green', alpha = 0.15)
# plt.grid()
# plt.xlabel('Количество обучающих образцов')
# plt.ylabel('Правильность')
# plt.legend(loc = 'upper left')
# plt.ylim([0.8, 1.03])
# plt.title("Кривая обучения")
# plt.tight_layout()
# plt.show()
#
# from sklearn.model_selection import validation_curve
#
#
# param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
# train_scores, test_scores = validation_curve(
#                 estimator=pile_lr,
#                 X=X_train,
#                 y=y_train,
#                 param_name='logisticregression__C',
#                 param_range=param_range,
#                 cv=10)
#
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)
#
# plt.plot(param_range, train_mean,
#          color='blue', marker='o',
#          markersize=5, label='Training accuracy')
#
# plt.fill_between(param_range, train_mean + train_std,
#                  train_mean - train_std, alpha=0.15,
#                  color='blue')
#
# plt.plot(param_range, test_mean,
#          color='green', linestyle='--',
#          marker='s', markersize=5,
#          label='Validation accuracy')
#
# plt.fill_between(param_range,
#                  test_mean + test_std,
#                  test_mean - test_std,
#                  alpha=0.15, color='green')
#
# plt.grid()
# plt.xscale('log')
# plt.legend(loc='lower right')
# plt.xlabel('Parameter C')
# plt.ylabel('Accuracy')
# plt.ylim([0.8, 1.0])
# plt.title("Правильность обучения")
# plt.tight_layout()
# # plt.savefig('images/06_06.png', dpi=300)
# plt.show()

# #НАСТРОЙКА ГИПЕРПАРАМЕТРОВ С ПОМОЩЬЮ РЕШЕТЧАТОГО ПОИСКА
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))
# param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
# param_grid = [{'svc__C': param_range,
#                'svc__kernel': ['linear']},
#               {'svc__C': param_range,
#                'svc__gamma': param_range,
#               'svc__kernel': ['rbf']}]
# gs = GridSearchCV(estimator= pipe_svc, param_grid = param_grid, scoring='accuracy',
#                   cv = 10, refit = True, n_jobs= 1)
# gs = gs.fit(X_train, y_train)
# print(gs.best_score_)
# print(gs.best_params_)
# clf = gs.best_estimator_
# clf.fit(X_train, y_train)
# print("Правильность: %.3f" % (clf.score(X_test, y_test)))
# #ВЛОЖЕННАЯ ПЕРЕКРЕСТНАЯ ПРОВЕРКА
# gs = GridSearchCV(estimator=pipe_svc, param_grid = param_grid, scoring='accuracy', cv =2)
# scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv = 5)
# print(np.mean(scores), np.std(scores))
# #В СРАВНЕНИИ С ДЕРЕВОМ
# from sklearn.tree import DecisionTreeClassifier
#
# gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
#                   param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
#                   scoring= 'accuracy',
#                   cv = 2)
# scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv = 5)
# print(np.mean(scores), np.std(scores))

# #РАБОТА С МАТРИЦАМИ НЕТОЧНОСТЕЙ
# from sklearn.metrics import confusion_matrix
#
# pipe_svc.fit(X_train, y_train)
# y_pred = pipe_svc.predict(X_test)
# confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
# print(confmat)
# fig, ax = plt.subplots(figsize = (2.5, 2.5))
# ax.matshow(confmat, cmap = plt.cm.Blues, alpha = 0.3)
# for i in range(confmat.shape[0]):
#     for j in range(confmat.shape[1]):
#         ax.text(x=j, y = i, s = confmat[i, j], va = 'center',
#                 ha = 'center')
# plt.xlabel("Спрогнозированная метка")
# plt.ylabel("Истинная метка")
# plt.tight_layout()
# plt.show()
#
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score, f1_score
#
# print("Точность: %.3f" % precision_score(y_true=y_test, y_pred=y_pred))
# print("Полнота: %.3f" % recall_score(y_true=y_test, y_pred=y_pred))
# print("Мера F1: %3.f" % f1_score(y_true=y_test, y_pred=y_pred))
#
# from sklearn.metrics import make_scorer, f1_score
# c_gamma_range = [0.01, 0.1, 1.0, 10.0]
# param_grid = [{'svc__C': c_gamma_range,
#                'svc__kernel': ['linear']},
#               {'svc__C': c_gamma_range,
#                'svc__gamma': c_gamma_range,
#                'svc__kernel': ['rbf']}]
# scorer = make_scorer(f1_score, pos_label = 0)
# gs = GridSearchCV(estimator=pipe_svc,
#                   param_grid=param_grid,
#                   scoring=scorer,
#                   cv = 10)
# gs = gs.fit(X_train, y_train)
# print(gs.best_score_)

# from sklearn.metrics import roc_curve, auc
# from scipy import __version__ as scipy_version
#
# from numpy import interp
#
# pipe_lr = make_pipeline(StandardScaler(),
#                         PCA(n_components=2),
#                         LogisticRegression(penalty='l2',
#                                            random_state=1,
#                                            solver='lbfgs',
#                                            C=100.0))
#
# X_train2 = X_train[:, [4, 14]]
#
# cv = list(StratifiedKFold(n_splits=3).split(X_train, y_train))
# print(cv)
# fig = plt.figure(figsize=(7, 5))
#
# mean_tpr = 0.0
# mean_fpr = np.linspace(0, 1, 100)
# all_tpr = []
#
# for i, (train, test) in enumerate(cv):
#     probas = pipe_lr.fit(X_train2[train],
#                          y_train[train]).predict_proba(X_train2[test])
#
#     fpr, tpr, thresholds = roc_curve(y_train[test],
#                                      probas[:, 1],
#                                      pos_label=1)
#     mean_tpr += interp(mean_fpr, fpr, tpr)
#     mean_tpr[0] = 0.0
#     roc_auc = auc(fpr, tpr)
#     plt.plot(fpr,
#              tpr,
#              label='ROC fold %d (area = %0.2f)'
#                    % (i + 1, roc_auc))
#
# plt.plot([0, 1],
#          [0, 1],
#          linestyle='--',
#          color=(0.6, 0.6, 0.6),
#          label='Random guessing')
#
# mean_tpr /= len(cv)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# plt.plot(mean_fpr, mean_tpr, 'k--',
#          label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
# plt.plot([0, 0, 1],
#          [0, 1, 1],
#          linestyle=':',
#          color='black',
#          label='Perfect performance')
#
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.legend(loc="lower right")
#
# plt.tight_layout()
# # plt.savefig('images/06_10.png', dpi=300)
# plt.show()

from sklearn.metrics import precision_score
from sklearn.metrics import make_scorer

pre_scorer = make_scorer(score_func=precision_score, pos_label = 1, greater_is_better=True, average ='micro')
print(pre_scorer)

X_imb = np.vstack((X[y==0], X[y==1][:40]))
y_imb = np.hstack((y[y==0], y[y==1][:40]))
print(X_imb.shape[0])#397, т.к к нему добавили 40 образцов 1 класса
y_pred = np.zeros(y_imb.shape[0])
print(np.mean(y_pred == y_imb) * 100)#проверка глупой модели, где перемножение на нулевую матрицу, дает предсказание для 0 класса
from sklearn.utils import resample
print("Начальное количество образцов класса 1:", X_imb[y_imb == 1].shape[0])
X_upsampled, y_upsampled = resample(X_imb[y_imb == 1], y_imb[y_imb == 1],#берем только значения 1 класса, 40 шт
                                    replace=True, #разрешаем дублирование
                                    n_samples=X_imb[y_imb == 0].shape[0], #хотим, чтобы количество 1 класса было равно 0 классу
                                    random_state= 123) #получаем на выходе 357 образцов
print("Конечное количество образцов класса 1: ", X_upsampled.shape[0])
X_bal = np.vstack((X[y==0], X_upsampled))
y_bal = np.hstack((y[y==0], y_upsampled))
y_pred = np.zeros(y_bal.shape[0])
print(np.mean(y_pred == y_bal)* 100)
