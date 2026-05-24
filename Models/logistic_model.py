from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle
import json 

class LogisticModel:
    def train_logistic(X_train,y_train,X_test,y_test):
        # print(X_train.shape)
        # print(X_test.shape)
        # print(y_train.shape)
        # print(y_test.shape)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train,y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        print("Logistic Regression Testing  Accuracy:", test_accuracy)
        print("Logistic Regression Training Accuracy:", train_accuracy)

        ##SAVING THE MODEL
        pickle.dump(model,open("Models/logistic_model.pkl","wb"))
        ## wb ==> write in binary mode
        
        ##Save metrics to Json
        metrics = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy
        }
        with open("Models/logistic_metrics.json","w")as f:
            json.dump(metrics,f)
        return model 