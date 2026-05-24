from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import json

class RandomForestModel:
    def train_random_forest(X_train, y_train, X_test, y_test):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        print("Random Forest Testing  Accuracy:", test_accuracy)
        print("Random Forest Training Accuracy:", train_accuracy)
        ##SAVING THE MODEL
        pickle.dump(model, open("Models/random_forest_model.pkl", "wb"))
        metrics = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy
        }
        with open("Models/random_forest_metrics.json","w") as f:
            json.dump(metrics,f)
        return model