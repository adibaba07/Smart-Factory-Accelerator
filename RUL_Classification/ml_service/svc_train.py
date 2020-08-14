import pandas as pd
import numpy as np
import sys
import json
import logging

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
from sklearn.preprocessing import MinMaxScaler
# from sklearn.externals import joblib
import joblib
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def get_logger(logname):
    # Create and configure logger
    logging.basicConfig(filename='app.log',
                        filemode='a',
                        format='%(name)s: %(levelname)s - %(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')

    # Creating an object
    logger = logging.getLogger(logname)

    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)
    return logger


def load_data(filepath1, filepath2, keys):

    cols_drop = ['op_setting3', 's1', 's5', 's6', 's10', 's14', 's16', 's18', 's19']
    data = pd.read_csv(filepath1, sep=" ", header=None)
    data.drop(data.columns[[-1, -2]], axis=1, inplace=True)
    data.columns = keys

    # grouped_data = pd.DataFrame(data.groupby('unit')['cycles'].max()).reset_index()
    # grouped_data.columns = ['unit','max']

    # df = data.merge(grouped_data, on=['unit'], how='left')
    data['RUL'] = data.groupby(['unit'])['cycles'].transform(max) - data['cycles']
    # df['RUL'] = df['max'] - df['cycles']
    # df.drop('max',axis=1,inplace=True)

    df_train = data.drop(cols_drop, axis=1)
    w1 = 35
    w0 = 25
    df_train['label1'] = np.where(df_train['RUL'] <= w1, 1, 0)
    df_train['label2'] = df_train['label1']
    df_train.loc[df_train['RUL'] <= w0, 'label2'] = 2

    y_true = pd.read_csv(filepath2, sep=" ", header=None)
    y_true.drop(y_true.columns[[1]], axis=1, inplace=True)
    y_true.columns = ['life']
    y_true = y_true.set_index(y_true.index + 1)

    return df_train, y_true


def scale(df):
    return (df - df.min())/(df.max()-df.min())


def main(config_path):
    """main entry point, load and validate config and call generate"""
    with open(config_path) as handle:
        config = json.load(handle)
        data_config = config.get("data", {})

        logger.debug("data config: {0}".format(data_config))

        file_path1 = data_config.get("file_path1")
        file_path2 = data_config.get("file_path2")
        columns = data_config.get("columns").split(",")

        df_train, y_truth = load_data(file_path1, file_path2, columns)
        print(df_train.shape) # (20631,20)
        print(y_truth.shape) # (100,1)

        # Min-Max Normalization
        # df_train['cycle_norm'] = df_train['cycles']
        for col in df_train.columns:
            if col[0] == 's' or col[0] == 'o':
                df_train[col] = scale(df_train[col])

        df_train = df_train.dropna(axis=1)
        # cols_normalize = df_train.columns.difference(['unit','cycles','RUL','label1','label2'])
        # min_max_scaler = MinMaxScaler()
        # norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(df_train[cols_normalize]),
        #                              columns=cols_normalize,
        #                              index=df_train.index)
        # join_df = df_train[df_train.columns.difference(cols_normalize)].join(norm_train_df)
        # train_df = join_df.reindex(columns=df_train.columns)
        # scaler_filename = "../ml_service/scaler_svc.joblib"
        # joblib.dump(min_max_scaler, scaler_filename)
        # print("Scaler file saved")
        print(df_train.head())

        features = df_train.columns.drop(['unit', 'cycles', 'RUL', 'label1', 'label2'])

        # Train-Val split
        X_train, X_val, Y_train, Y_val = train_test_split(df_train[features], df_train['label1'], test_size=0.05,
                                                          shuffle=False, random_state=42)

        print("Train_shape: " + str(X_train.shape)) # (19599, 15)
        print("Val_shape: " + str(X_val.shape)) # (1032, 15)
        print("No of positives in train: " + str(Y_train.sum())) # 3420
        print("No of positives in val: " + str(Y_val.sum())) # 180

        # Training SVM Classfier
        print("Start Training...")

        # defining parameter range
        # param_grid = {'C': [i for i in range(1,10)], 'gamma': ['auto'], 'kernel': ['rbf'], 'degree':[3]}
        #
        # grid = GridSearchCV(SVC(), param_grid, verbose=True)
        #
        # # fitting the model for grid search
        # grid.fit(X_train, Y_train)
        #
        # # print best parameter after tuning
        # print(grid.best_params_)
        #
        # # print how our model looks after hyper-parameter tuning
        # print(grid.best_estimator_)

        # result of CV
        # {'C': 8, 'degree': 3, 'gamma': 'auto', 'kernel': 'rbf'}
        # SVC(C=8, cache_size=200, class_weight=None, coef0=0.0,
        #     decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
        #     max_iter=-1, probability=False, random_state=None, shrinking=True,
        #     tol=0.001, verbose=False)

        # Model SVC
        clf = SVC(C=8.0, kernel='rbf', degree=3, gamma='auto', shrinking=True, verbose=True, max_iter=-1,
                  random_state=42)
        clf.fit(X_train, Y_train)

        print("Validation Accuracy: " + str(accuracy_score(Y_val, clf.predict(X_val))))
        # [LibSVM]Validation Accuracy: 0.9496124031

        # training metrics
        pred_train = clf.predict(df_train[features])
        pred_train = np.where(pred_train > 0.5, 1, 0)
        print('Accuracy: {}'.format(accuracy_score(df_train['label1'], pred_train)))
        # Accuracy: 0.9544374969

        print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
        cm = confusion_matrix(df_train['label1'], pred_train)
        print(cm)
        # Confusion matrix
        # - x-axis is true labels.
        # - y-axis is predicted labels
        # [[16704   327]
        #  [  613  2987]]

        # save model
        joblib.dump(clf, "svc_rul.joblib")
        # joblib.dump(clf, "svc_rul.joblib")
        print("Model saved")


if __name__ == '__main__':
    logger = get_logger(__name__)
    if len(sys.argv) < 2:
        logger.error("Please provide configuration file path.")
    else:
        main(sys.argv[1])
