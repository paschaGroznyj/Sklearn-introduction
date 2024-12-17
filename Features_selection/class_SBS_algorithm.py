from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class SBS():
    def __init__(self, estimator, k_features, scoring = accuracy_score, test_size = 0.25, random_state = 1):
        self.scoring = scoring #метрика оценки правильности обучения
        self.estimator = clone(estimator)#передаем клон обученной модели
        self.k_features = k_features#количество признаков, которое мы хотим оставить
        self.test_size = test_size
        self.random_state = random_state
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state=self.random_state)
        dim = X_train.shape[1]#13
        self.indices_ = tuple(range(dim))#создаем кортеж значений(0, 1, 2...12)
        self.subsets_ = [self.indices_]#сохраняем нашу комбинацию
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)#считаем точность на первом проходе
        self.scores_ = [score]#записываем первый результат точности

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):#функция combinations комбинирует кортеж, с длиной комбинаций r
                score = self._calc_score(X_train, y_train, X_test, y_test, p)#p это кортеж комбинаций
                scores.append(score)
                subsets.append(p)
            best = np.argmax(scores)#получаем индекс топ точности
            self.indices_ = subsets[best]#берем лучшую комбинацию для точности
            self.subsets_.append(self.indices_)#сохраняем комбинацию
            dim -= 1
            self.scores_.append(scores[best])#сохраняем значение точности
        self.k_score = self.scores_[-1]
        print(self.scores_)
        return self
    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)#можно спокойно передать кортеж, вместо среза
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
