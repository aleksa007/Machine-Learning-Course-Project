import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from termcolor import colored as cl

data = pd.read_csv ('qsar_fish_toxicity.csv')  #Loading data
#data = data.drop(['Unnamed: 0'], axis = 1)
print(cl(data.dtypes, attrs = ['bold']))
X=data.drop(["response"], axis =1)  #Droppping target variablr
y=data["response"]  #Extract targer var


sns.distplot(y)  #Using seaborn pcket to check if data is noramlly distributed
plt.show()




plt.hist(y)                         #histogram of target veravle (RESPONSE)
plt.title("Response Variable")
plt.ylabel("Frequency")
plt.xlabel("Variables")
plt.show()




columns=['CIC0','SM1_Dz(Z)','GATS1i','NdsCH','NdssC','MLOGP']  #Creating colomns lsit to be able to add them in loop

fig, axs = plt.subplots(3,2, figsize=(15, 6), facecolor='w', edgecolor='k') #Figure created
fig.subplots_adjust(hspace = 1, wspace=0.6)
fig.suptitle(' 6 features vary with the target variable')

axs = axs.ravel()

for i in range(6):   #Using loop to add each subplot in big polot

    axs[i].scatter(X.iloc[:,i], y)
    axs[i].set_title(str(columns[i]) + " vs Target Variable")
    axs[i].set_xlabel("response")
    axs[i].set_ylabel(str(columns[i]))

plt.show()


corrMatrix = data.corr() #Correlation matrix

sns.heatmap(data.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
plt.show()

print (corrMatrix)



def preprocess():
    data = pd.read_csv ('qsar_fish_toxicity.csv')

    X=data.drop(["response"], axis =1)  #Droppping target variablr
    y=data["response"]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    return X_train, X_test, y_train, y_test


def lin_model(X_train, X_test, y_train, y_test):
    lm = LinearRegression()
    model = lm.fit(X_train, y_train)
    predictions = lm.predict(X_test)
    print("Test set score: {:.2f}".format(lm.score(X_test, y_test)))
    print ("Score:", model.score(X_test, y_test))
    print ("r**2:", r2_score(y_test, predictions))


    plt.figure()
    plt.scatter(y_test, predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.show()

def ridge(X_train, X_test, y_train, y_test):
    ridge = Ridge(alpha=1).fit(X_train, y_train)
    predictions = ridge.predict(X_test)
    print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))
    print ("Score:", ridge.score(X_test, y_test))
    print ("r**2:", r2_score(y_test, predictions))

def lasso_cheker(X_train, X_test, y_train, y_test):
    """
    Testing which hyperparameter will give best accuracy_score
    """
    lasso = Lasso(alpha=1, max_iter=100000).fit(X_train, y_train)
    print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
    print("Number of features used:", np.sum(lasso.coef_ != 0))

    lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
    print("alpha = 0.01: Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
    print("alpha = 0.01: Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
    print("alpha = 0.01: Number of features used:", np.sum(lasso001.coef_ != 0))

    lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
    print("alpha = 0.0001: Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
    print("alpha = 0.0001: Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
    print("alpha = 0.0001: Number of features used:", np.sum(lasso00001.coef_ != 0))


def ridge_cheker(X_train, X_test, y_train, y_test):
    """
    Testing which hyperparameter will give best accuracy_score
    """
    ridge = Ridge(alpha=1, max_iter=100000).fit(X_train, y_train)
    print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))
    print("Number of features used:", np.sum(ridge.coef_ != 0))

    ridge001 = Ridge(alpha=0.01, max_iter=100000).fit(X_train, y_train)
    print("alpha = 0.01: Training set score: {:.2f}".format(ridge001.score(X_train, y_train)))
    print("alpha = 0.01: Test set score: {:.2f}".format(ridge001.score(X_test, y_test)))
    print("alpha = 0.01: Number of features used:", np.sum(ridge001.coef_ != 0))

    ridge00001 = Ridge(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
    print("alpha = 0.0001: Training set score: {:.2f}".format(ridge00001.score(X_train, y_train)))
    print("alpha = 0.0001: Test set score: {:.2f}".format(ridge00001.score(X_test, y_test)))
    print("alpha = 0.0001: Number of features used:", np.sum(ridge00001.coef_ != 0))

def ridge(X_train, X_test, y_train, y_test):
    ridge001 = Ridge(alpha=0.01, max_iter=100000).fit(X_train, y_train)
    predictions=ridge001.predict(X_test)
    print("Training set score {}:".format(ridge001.score(X_train, y_train)))
    print("Test set score {}: ".format(ridge001.score(X_test, y_test)))
    print ("r**2:", r2_score(y_test, predictions))


def sv(X_train, X_test, y_train, y_test):  #SVR CLASSIFIER WITH CROSS VALL
    svr = SVR(C = 100, epsilon = 0.0001, gamma = 0.0001)

    svr.fit(X_train, y_train)

    svr_pred = svr.predict(X_test)
    accuracy = cross_val_score(svr, X_test, y_test, cv=10,scoring='r2')


    print("Training set score: {:.3f}".format(svr.score(X_train, y_train)))
    print("Test set score: {:.3f}".format(svr.score(X_test, y_test)))
    print("Validation set score: {:.3f}".format(svr.score(X_test, y_test)))

    #print(accuracy)
    print("cross validation score", np.round(accuracy.mean(),3))


def DTR(X_train, X_test, y_train, y_test): #DecisionTreeRegressor With Cross Vall
    dtr = DecisionTreeRegressor()

    dtr.fit(X_train, y_train)

    dtr_pred = dtr.predict(X_test)
    accuracy = cross_val_score(dtr, X_test, y_test, cv=10,scoring='r2')


    print("Training set score: {:.3f}".format(dtr.score(X_train, y_train)))
    print("Test set score: {:.3f}".format(dtr.score(X_test, y_test)))
    print("Validation set score: {:.3f}".format(dtr.score(X_test, y_test)))

    #print(accuracy)
    print("cross validation score", np.round(accuracy.mean(),3))


def GridDTR(X_train, X_test, y_train, y_test): #DecisionTreeRegressor  Grid Seatch
    scoring = make_scorer(r2_score)
    g_cv = GridSearchCV(DecisionTreeRegressor(random_state=0),
              param_grid={'min_samples_split': range(2, 10)},
              scoring=scoring, cv=5, refit=True)

    g_cv.fit(X_train, y_train)
    predict=g_cv.predict(X_test)
    #g_cv.best_params_

    result = g_cv.cv_results_
    #print(result)
    print("r2 score: ", r2_score(y_test,predict))

def GridSVR(X_train, X_test, y_train, y_test):
    parameters = {'kernel': ('linear', 'rbf','poly'), 'C':[1.5, 10],'gamma': [1e-7, 1e-4],'epsilon':[0.1,0.2,0.5,0.3]}
    svr = SVR()
    clf = GridSearchCV(svr, parameters)
    clf.fit(X_train,y_train)
    predict=clf.predict(X_test)

    print("r2 score: ", r2_score(y_test,predict))





true_data = pd.read_csv('qsar_fish_toxicity.csv')
miss_data = pd.read_csv('2024812_qsar.csv')
miss_data = miss_data.drop(['Unnamed: 0'], axis = 1)

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.metrics import mean_squared_error


sim_imp = SimpleImputer(strategy='mean').fit(miss_data)
simple_impute_data = sim_imp.transform(miss_data)

print("SimpleImputer",mean_squared_error(true_data, simple_impute_data))

knn_imp = KNNImputer(n_neighbors = 4).fit(miss_data)
knn_impute_data = knn_imp.transform(miss_data)

print("KNN Imputer",mean_squared_error(true_data, knn_impute_data))

iter_imp = IterativeImputer(max_iter=50, random_state=0).fit(miss_data)
iter_impute_data = iter_imp.transform(miss_data)

print("IterativeImputer",mean_squared_error(true_data, iter_impute_data))






#Same rules for running like in classification part
