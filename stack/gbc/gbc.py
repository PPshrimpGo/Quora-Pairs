from time import time
import datetime
import pandas as pd
import numpy as np

# get train data and test data
def load_data():
    now = datetime.datetime.now()
    print ("load data start in " + now.strftime('%Y-%m-%d %H:%M:%S'))

    train_df = pd.read_csv('../train_fold_all.csv')
    test_df = pd.read_csv('../test_fold_all.csv')
    train_df = train_df.fillna(0.0)
    test_df = test_df.fillna(0.0)
    col = [c for c in train_df.columns if c[:1] == 'z' or c[:1] == 'f']
    X_train = train_df[col]
    y_train = train_df['is_duplicate']
    X_test = test_df[col]
    print (np.shape(X_train))
    print (np.shape(y_train))
    print (np.shape(X_test))

    now = datetime.datetime.now()
    print ("load data done in " + now.strftime('%Y-%m-%d %H:%M:%S'))

    return X_train, y_train, X_test

# StandardScalar
def StandardScalar(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test

# LogisticRegression model
def LogisticRegression(X_train, y_train, X_test):
    from sklearn.linear_model import LogisticRegression
    now = datetime.datetime.now()
    print ("logestic regression start in " + now.strftime('%Y-%m-%d %H:%M:%S'))

    LR = LogisticRegression(solver='sag',
                            penalty='l2',
                            class_weight=None,
                            C=1.2)
    LR.fit(X_train, y_train)
    now = datetime.datetime.now()
    print ("logestic regression train done in " + now.strftime('%Y-%m-%d %H:%M:%S'))

    y_pred_LR = LR.predict_proba(X_test)
    y_pred_LR = pd.DataFrame(y_pred_LR[:,1:2],columns=['LR_predictions'])
    y_pred_LR.to_csv('LR_result_1.csv', index=False)
    now = datetime.datetime.now()
    print ("logestic regression predict done in " + now.strftime('%Y-%m-%d %H:%M:%S'))

# RandomForestClassifier model
def RandomForestClassifier(X_train, y_train, X_test):
    from sklearn.ensemble import RandomForestClassifier
    now = datetime.datetime.now()
    print ("RandomForestClassifier start in " + now.strftime('%Y-%m-%d %H:%M:%S'))
    RFC = RandomForestClassifier(max_features='auto',
                                 max_depth=8,
                                 random_state=10,
                                 n_estimators=200,
                                 n_jobs = 4)
    RFC.fit(X_train, y_train)
    now = datetime.datetime.now()
    print ("RandomForestClassifier train done in " + now.strftime('%Y-%m-%d %H:%M:%S'))

    y_pred_RFC = RFC.predict_proba(X_test)
    y_pred_RFC = pd.DataFrame(y_pred_RFC[:,1:2],columns=['RFC_predictions'])
    y_pred_RFC.to_csv('RFC_result_1.csv', index=False)
    now = datetime.datetime.now()
    print ("RandomForestClassifier predict done in " + now.strftime('%Y-%m-%d %H:%M:%S'))


def GradientBoostingClassifier(X_train, y_train, X_test):
    from sklearn.ensemble import GradientBoostingClassifier
    now = datetime.datetime.now()
    print ("GradientBoostingClassifier start in " + now.strftime('%Y-%m-%d %H:%M:%S'))
    GBC = GradientBoostingClassifier(max_features = 'sqrt',
                                     n_estimators = 300,
                                     learning_rate = 0.02,
                                     max_depth = 8,
                                     subsample = 0.8,
                                     n_jobs =4)
    GBC.fit(X_train, y_train)
    now = datetime.datetime.now()
    print ("GradientBoostingClassifier train done in " + now.strftime('%Y-%m-%d %H:%M:%S'))

    y_pred_GBC = GBC.predict_proba(X_test)
    y_pred_GBC = pd.DataFrame(y_pred_GBC[:,1:2],columns=['GBC_predictions'])
    y_pred_GBC.to_csv('GBC_result_all.csv', index=False)
    now = datetime.datetime.now()
    print ("GradientBoostingClassifier predict done in " + now.strftime('%Y-%m-%d %H:%M:%S'))


def MLPClassifier(X_train, y_train):
    from sklearn.neural_network import MLPClassifier

    now = datetime.datetime.now()
    print ("MLPClassifier start in " + now.strftime('%Y-%m-%d %H:%M:%S'))
    MLPC = MLPClassifier()
    MLPC.fit(X_train, y_train)
    now = datetime.datetime.now()
    print ("MLPClassifier train done in " + now.strftime('%Y-%m-%d %H:%M:%S'))

    y_pred_MLPC = MLPC.predict_proba(X_test)
    y_pred_MLPC = pd.DataFrame(y_pred_MLPC[:,1:2],columns=['MLPC_predictions'])
    y_pred_MLPC.to_csv('MLPC_result.csv', index=False)
    now = datetime.datetime.now()
    print ("MLPClassifier predict done in " + now.strftime('%Y-%m-%d %H:%M:%S'))
# SVC model
def SVC(X_train, y_train, X_test):
    from sklearn.svm import SVC
    now = datetime.datetime.now()
    print("SVC start in " + now.strftime('%Y-%m-%d %H:%M:%S'))

    svc = SVC(class_weight='balanced',
              kernel='rbf')
    svc.fit(X_train, y_train)
    now = datetime.datetime.now()
    print("SVC train done in " + now.strftime('%Y-%m-%d %H:%M:%S'))

    y_pred_SVC = svc.predict_proba(X_test)
    y_pred_SVC = pd.DataFrame(y_pred_SVC[:, 1:2], columns=['SVC_predictions'])
    y_pred_SVC.to_csv('SVC_result.csv', index=False)
    now = datetime.datetime.now()
    print("SVC predict done in " + now.strftime('%Y-%m-%d %H:%M:%S'))

if __name__ == '__main__':
    X_train, y_train, X_test = load_data()
    X_train, X_test = StandardScalar(X_train, X_test)

    #LogisticRegression(X_train, y_train, X_test)
    GradientBoostingClassifier(X_train, y_train, X_test)
    #SVC(X_train, y_train, X_test)