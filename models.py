from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

def baseline(x_train, x_test, y_train, y_test):

    logreg = LogisticRegression().fit(x_train, y_train)
    scores = cross_val_score(logreg, x_train, y_train)
    y_pred = logreg.predict(x_test)
    print("Logistic Regression Training set score: {:.3f}".format(logreg.score(x_train, y_train)))
    print("Logistic Regression Test set score: {:.3f}".format(logreg.score(x_test, y_test)))

    print("Cross validation scores: \n", scores)

    cr = classification_report(y_test, y_pred)
    return cr