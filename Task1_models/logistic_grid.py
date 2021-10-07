import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# reference: https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/
    # parameter in the logistic parameter model ->
    # search for C only, the solver and the penalty has been searched already -> takes almost 20 minutes
def grid_search_logistic(train_features, train_labels):
    model = LogisticRegression()
    parameter = dict()
    parameter['C'] = np.linspace(0.0001, 100, 20)
    grid_search = GridSearchCV(model, parameter, scoring='f1_micro', n_jobs=-1) # f1_micro
    grid_search.fit(train_features, train_labels)

    return  grid_search.best_params_['C'], grid_search.best_score_

