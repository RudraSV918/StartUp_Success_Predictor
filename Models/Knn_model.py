from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
import json

class KnnModel:
    def train_knn(X_train, y_train, X_test, y_test):
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        print("KNN Testing  Accuracy:", test_accuracy)
        print("KNN Training Accuracy:", train_accuracy)

        ##SAVING THE MODEL
        pickle.dump(model, open("Models/knn_model.pkl", "wb"))
        ## wb ==> write in binary mode

        ##save metrics to Json
        metrics = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy
        }
        with open("Models/knn_metrics.json","w") as f:
            json.dump(metrics,f)
        return model