import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def grid_search_rforest(train_features, train_labels):
    '''Grid search for the random forest
       return the parameters and its f1 score
    '''
    n_estimators = [10, 100, 1000]
    max_features = ['sqrt', 'log2']
    grid = dict(n_estimators=n_estimators,max_features=max_features)
    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=grid, n_jobs=-1, scoring='f1_micro',error_score=0)
    grid_search.fit(train_features, train_labels)

    return grid_search.best_params_, grid_search.best_score_