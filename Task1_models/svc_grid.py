import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC,LinearSVC

def grid_search_svc(train_features, train_labels):
    kernel = ['poly', 'rbf', 'sigmoid']
    C = [50, 10, 1.0, 0.1, 0.01]
    gamma = ['scale']
    grid = dict(kernel=kernel,C=C,gamma=gamma)
    grid_search = GridSearchCV(estimator=SVC(), param_grid=grid, n_jobs=-1, scoring='f1_micro',error_score=0)
    grid_search.fit(train_features, train_labels)

    return grid_search.best_params_, grid_search.best_score_

# SVCLinear perform a litte worse but much faster than SVC
def grid_search_svclinear(train_features, train_labels):
    penalty = ['l1','l2']
    loss = ['hinge','squared_hinge']
    grid = dict(penalty=penalty,loss=loss)
    grid_search = GridSearchCV(estimator=LinearSVC(), param_grid=grid, n_jobs=-1, scoring='f1_micro',error_score=0)
    grid_search.fit(train_features, train_labels)

    return grid_search.best_params_, grid_search.best_score_