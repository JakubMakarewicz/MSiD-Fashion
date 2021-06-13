from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def get_classifier(x_train, y_train, neighbors, algorithm='auto'):
    model = KNeighborsClassifier(neighbors, algorithm=algorithm)
    model.fit(x_train, y_train)
    return model


def test_classifier(x_test, y_test, model):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy