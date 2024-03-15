from dataset.Dataset import prepare_dataset
from sklearn.ensemble import RandomForestClassifier as rfc


X_train, y_train, X_test, y_test = prepare_dataset()
model = rfc().fit(X_train, y_train)
preds = model.predict(X_test)
print("Model Accuracy: ", np.mean(y_test == preds))

