import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataPreprocessing:

    def load_and_preprocess():
        df = pd.read_csv("Data/startup_dataset.csv")
        # print(df.head())

        # print("BEFORE ENCODING")
        # print(df["IndustryType"][:3])
        # print(df["FounderEducation"][:3])
        # print(df["ProductStage"][:3])

        le = LabelEncoder()
        df["IndustryType"] = le.fit_transform(df["IndustryType"])
        df["FounderEducation"] = le.fit_transform(df["FounderEducation"])
        df["ProductStage"] = le.fit_transform(df["ProductStage"])

        # print("\nAFTER ENCODING")
        # print(df["IndustryType"][:3])
        # print(df["FounderEducation"][:3])
        # print(df["ProductStage"][:3])

        X = df.drop("StartupSuccess",axis=1)
        y = df["StartupSuccess"]

        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        return X_train,X_test,y_train,y_test
    