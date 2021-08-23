from preprocess import preprocess
from models import baseline

def run():
    x_train, x_test, y_train, y_test = preprocess()

    log_reg_score = baseline(x_train, x_test, y_train, y_test)

    print("Logistic regression Scores: \n", log_reg_score)

    return None

if __name__ == '__main__':
    run()