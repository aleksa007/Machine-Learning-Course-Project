import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess():
    data = pd.read_csv('../data/2024812_mocap.csv').drop(['Unnamed: 0'], axis = 1)

    X = data.drop(["Class"], axis=1)
    y = data.Class
    x_train, x_test, y_train, y_test = train_test_split(X, y)

    return x_train, x_test, y_train, y_test
