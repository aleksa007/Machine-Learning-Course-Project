

# coding: utf-8
#get_ipython().run_line_magic('matplotlib', 'inline')
#import mglearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

#load data 1.1
data = pd.read_csv ('2024812_mocap.csv')  #Importing dataset
data = data.drop(['Unnamed: 0'], axis = 1) #Dropping unnamed column


#1.2
data_points=data.Class.value_counts() #Number of data points for each class


def make_subplots(cl: int, data=data):
    """
    Fucntion for making subplots for each class
    """

    class_data=data[data.Class == cl].iloc[[-1]].T #Transforimng data

    X = class_data.iloc[1:13] #Indexes
    Y = class_data.iloc[13:25]
    Z = class_data.iloc[25:]

    fig = plt.figure(figsize=(10,8)) #Creating figure
    fig.suptitle('4 subplots for Class {}'.format(cl)) #Setting title

    ax = fig.add_subplot(2, 2, 1)

    ax.scatter(X, Y)
    ax.set_xlabel("Y")
    ax.set_ylabel("X")
    ax.set_title("X vs Y")
    ax.grid(True)


    ax = fig.add_subplot(2, 2, 2)

    ax.scatter(X, Z)
    ax.set_xlabel("Z")
    ax.set_ylabel("X")
    ax.set_title("X vs Z")
    ax.grid(True)

    ax = fig.add_subplot(2, 2, 3)

    ax.scatter(Y, Z)
    ax.set_xlabel("Z")
    ax.set_ylabel("Y")
    ax.set_title("Y vs Z")
    ax.grid(True)

    surf = fig.add_subplot(2, 2, 4, projection='3d')

    surf.scatter(X, Y, Z)
    surf.set_xlabel("X")
    surf.set_ylabel("Y")
    surf.set_zlabel("Z")
    surf.set_title("X vs Y vs Z")
    surf.grid(True)


    plt.show()

#To run the code uncomment comment below and enter class number in brackets
#make_subplots(2)
#make_subplots(4)
#make_subplots(5)

#1.3 Classificitaion task


def preprocess():
    """
    preprocess function used to run all models in this script
    """
    data = pd.read_csv('2024812_mocap.csv').drop(['Unnamed: 0'], axis = 1)

    X = data.drop(["Class"], axis=1)
    y = data.Class
    x_train, x_test, y_train, y_test = train_test_split(X, y)
    x_train=preprocessing.scale(x_train)
    x_test=preprocessing.scale(x_test)


    return x_train, x_test, y_train, y_test


def baseline(x_train, x_test, y_train, y_test):
    """
    Baseline Classifier
    """

    logreg = LogisticRegression().fit(x_train, y_train)
    #scores = cross_val_predict(logreg, x_train, y_train)
    y_pred = logreg.predict(x_test)
    #y_pred_CV=cross_val_predict(x_train,y_train)
    #print("Logistic Regression Training set score: {:.3f}".format(logreg.score(x_train, y_train)))
    #print("Logistic Regression Test set score : {:.3f}".format(logreg.score(x_test, y_test)))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))  #0.79
    #print("Cross validation score:", metrics.accuracy_score(y_test,y_pred_CV))


    cr = classification_report(y_test, y_pred)
    return (cr)

def cross_val(x_train, x_test, y_train, y_test):
    X = data.drop(["Class"], axis=1)
    y = data.Class
    logreg=LogisticRegression()
    scores=cross_val_score(logreg,x_train,y_train)
    print ("CV score: {}".format(np.mean(scores)))


#Alghorithms used to beat baseline calssifier

def KNN(x_train, x_test, y_train, y_test):
    model=KNeighborsClassifier(n_neighbors=3).fit(x_train,y_train)
    predict=model.predict(x_test)
    print(("Accuracy:",metrics.accuracy_score(y_test, predict)))
    cr = classification_report(y_test, predict)
    return (cr) #0.848

def SVM(x_train, x_test, y_train, y_test):
    model=svm.SVC(kernel="linear").fit(x_train,y_train)
    predict=model.predict(x_test)
    print(("Accuracy:",metrics.accuracy_score(y_test, predict)))
    cr = classification_report(y_test, predict)
    return (cr) #0.78

def MLP(x_train, x_test, y_train, y_test):
    mlp = MLPClassifier()
    mlp.fit(x_train,y_train)
    predict_train = mlp.predict(x_train)
    predict_test = mlp.predict(x_test)
    print(("Train Accuracy:",metrics.accuracy_score(y_train, predict_train)))
    print(("Test Accuracy :",metrics.accuracy_score(y_test, predict_test)))
    cr_train = classification_report(y_train, predict_train)
    print(cr_train) #0.92
    cr_test = classification_report(y_test, predict_test)
    return (cr_test)

def PcA(x_train, x_test, y_train, y_test):
    pca = PCA()
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    pca.fit(x_train, y_train)

    classifier = RandomForestClassifier(max_depth=2, random_state=0) #to make predictions
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    print('Accuracy', accuracy_score(y_test, y_pred))
    print ("Report: ", classification_report(y_test, y_pred))

#hyperparameter tunings

def Hype_LogReg(x_train, x_test, y_train, y_test):
    dual=[True,False]
    max_iter=[100,110,120,130,140]
    param_grid = dict(dual=dual,max_iter=max_iter)
    lr = LogisticRegression(penalty='l2')
    grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv = 3, n_jobs=-1)


    grid_result = grid.fit(x_train, y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_estimator_))

def SVM1_Hype(x_train, x_test, y_train, y_test): #It is not reccomened to run these tuner due to high processing time
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    grid_search.fit(x_train, y_train)

    """
    SVC(C=10, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    """

    print("Best estimator:\n{}".format(grid_search.best_estimator_))

def MLP_Hype(x_train, x_test, y_train, y_test):
    mlp = MLPClassifier(max_iter=100)
    parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],}
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    clf.fit(x_train,y_train)
    print('Best parameters found:\n', clf.best_params_)



def Knn_Hype(x_train, x_test, y_train, y_test):

    leaf_size = list(range(1,50))
    n_neighbors = list(range(1,30))
    p=[1,2]

    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

    knn_2 = KNeighborsClassifier()

    clf = GridSearchCV(knn_2, hyperparameters, cv=10)

    best_model = clf.fit(x_train,y_train)

    print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
    print('Best p:', best_model.best_estimator_.get_params()['p'])
    print('Best n_neighbors:', best_model.best_estimator_)

def ensemble(x_train, x_test, y_train, y_test):  #Model trainging with ensemble learning
    # first we scale the data,
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    # transform data
    X_train = scaler.transform(x_train)
    X_test = scaler.transform(x_test)

    model1 = LogisticRegression()
    model2 = SVC()
    model3= KNeighborsClassifier(n_neighbors=3)
    model4 = MLPClassifier()
    model5= PCA()


    model1.fit(X_train,y_train)
    model2.fit(X_train,y_train)
    model3.fit(X_train,y_train)
    model4.fit(X_train,y_train)
    model4.fit(X_train,y_train)

    pred1=model1.predict(X_test)
    pred2=model2.predict(X_test)
    pred3=model2.predict(X_test)
    pred4=model2.predict(X_test)
    pred5=model2.predict(X_test)



    final_pred = np.array([])
    for i in range(0,len(X_test)):
        m = stats.mode([pred1[i], pred2[i]])
        final_pred = np.append(final_pred, m[0])

    print("Accuracy: ", np.mean(final_pred - y_test == 0))
    print ("Report: ", classification_report(final_pred,y_test))
    #For each model use following code print(classification_report(pred(num),y_test))


print(data_points)



def run():
    """
    To run any model type function name and watch :)
    """

    x_train, x_test, y_train, y_test = preprocess()

    model = ensemble(x_train, x_test, y_train, y_test)

    print(model)

    return None


if __name__ == '__main__':
    run()
