import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

def grid_search_knn(train_features, train_labels):
    n_neightbors = range(1,21,2)
    weight = ['uniform', 'distance']
    metric = ['euclidean', 'manhattan', 'minkowski']
    grid = dict(n_neighbors=n_neightbors,weights=weight,metric=metric)
    grid_search = GridSearchCV(estimator=KNeighborsClassifier(),param_grid=grid,scoring='f1_micro',error_score=0)
    grid_search.fit(train_features, train_labels)
    return grid_search.best_params_, grid_search.best_score_