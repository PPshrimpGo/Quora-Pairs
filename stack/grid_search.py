from time import time
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import datetime

# get train data and test data
def load_data():
    now = datetime.datetime.now()
    print ("load data start in " + now.strftime('%Y-%m-%d %H:%M:%S'))

    train_label = pd.read_csv('./train_fold_all.csv')[:]

    train_df = pd.read_csv('./train_fold_all.csv')[:]
    train_df = train_df.fillna(0.0)

    col = [c for c in train_df.columns if c[:1] == 'z' or c[:1] == 'f']
    X_train = train_df[col]
    y_train = train_label['is_duplicate']
    # X_test = test_df[col]
    print (np.shape(X_train))
    print (np.shape(y_train))
    # print (np.shape(X_test))

    now = datetime.datetime.now()
    print ("load data done in " + now.strftime('%Y-%m-%d %H:%M:%S'))

    return X_train, y_train

# StandardScalar
def StandardScalar(X_train):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    # X_test = sc.transform(X_test)
    return X_train

def LogisticRegression(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    parameters = {
        'C':[0.6, 0.8, 1.0, 1.2],
        'class_weight':[None, 'balanced'],
    }

    LR = LogisticRegression()
    grid_search = GridSearchCV(estimator=LR, param_grid=parameters, cv=5, scoring='neg_log_loss',n_jobs=4)

    now = datetime.datetime.now()
    print ("logestic regression grid_search start in " + now.strftime('%Y-%m-%d %H:%M:%S'))

    grid_search.fit(X_train, y_train)
    print ("logestic regression grid_search done in " + now.strftime('%Y-%m-%d %H:%M:%S'))

    results = grid_search.grid_scores_
    for result in results:
        print(result)
    print("\nBest score: %0.3f\n" % grid_search.best_score_)
    print ("---------best parameters---------")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print ("%s: %r" % (param_name, best_parameters[param_name]))

def RandomForestClassifier(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    parameters = {
        'n_estimators':[8, 20],
        'max_depth':[6, 8, 10],
        # 'class_weight': [None, 'balanced'],
        'max_features':['auto','sqrt','log2'],
    }

    rfc = RandomForestClassifier()

    grid_search = GridSearchCV(estimator=rfc, param_grid=parameters, cv=5, scoring='neg_log_loss', n_jobs=10)

    now = datetime.datetime.now()
    print("RandomForestClassifier grid_search start in " + now.strftime('%Y-%m-%d %H:%M:%S'))

    grid_search.fit(X_train, y_train)
    print("RandomForestClassifier grid_search done in " + now.strftime('%Y-%m-%d %H:%M:%S'))

    results = grid_search.grid_scores_
    for result in results:
        print(result)
    print("\nBest score: %0.3f\n" % grid_search.best_score_)
    print("---------best parameters---------")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

if __name__ == '__main__':
    X_train, y_train = load_data()
    X_train = StandardScalar(X_train)

    # LogisticRegression(X_train, y_train)
    RandomForestClassifier(X_train, y_train)