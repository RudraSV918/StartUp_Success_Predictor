from sklearn.preprocessing import LabelEncoder

labels = ["cat","dog","dog","cat","chakli"]

le=LabelEncoder()

encoded = le.fit_transform(labels)
print(encoded)


print(le.inverse_transform(encoded))