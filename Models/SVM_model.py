from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import json

class SVMModel:
    def train_svm(self, X_train, X_test, y_train, y_test):
        ##pipeline
        model = Pipeline([
            ("scaler",StandardScaler()),
            ("svm", SVC(
                kernel = "rbf",
                probability = True
            ))
        ])
        ##prediction
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        ##testing and training accuracy
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        print("SVM Testing  Accuracy:", test_accuracy)
        print("SVM Training Accuracy:", train_accuracy)
        ##saving the model
        pickle.dump(model, open("Models/svm_model.pkl", "wb"))
        metrics = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy
        }
        with open("Models/svm_metrics.json","w") as f:
            json.dump(metrics,f)

        return model
