##this file will train and save models

from Dataset.preprocessing import DataPreprocessing
from Models.Knn_model import KnnModel
from Models.logistic_model import LogisticModel 
from Models.RandomForest_model import RandomForestModel


def main():
    # print("Loading and preprocessing dataset")
    X_train,X_test,y_train,y_test = DataPreprocessing.load_and_preprocess()
    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)
    print("Training Logistic Regression model")
    Lrmodel = LogisticModel.train_logistic(X_train,y_train,X_test,y_test)

    print("\n training KNN MODEL")
    knn_model = KnnModel.train_knn(X_train,y_train,X_test,y_test)

    print("\n training Random Forest MODEL")
    rf_model = RandomForestModel.train_random_forest(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()