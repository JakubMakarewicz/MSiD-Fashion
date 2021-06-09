from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def get_classifier(x_train, y_train):
    model = GaussianNB()
    model.fit(x_train, y_train)
    return model


def test_classifier(x_test, y_test, model):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

