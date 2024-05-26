import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score

CSV_PATH = "C:/Users/efefa/Desktop/Python-Proje/Data/features_30_sec.csv"
JSON_PATH = "C:/Users/efefa/Desktop/Python-Proje/Data/features_30_sec.json"

def csv_to_json(csv_path, json_path):
    df = pd.read_csv(csv_path)
    data = {
        "mfcc": df.drop(columns=['filename', 'label']).values.tolist(),
        "label": df['label'].values.tolist()
    }
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file)
    return json_path

# CSV dosyasını JSON'a dönüştür
csv_to_json(CSV_PATH, JSON_PATH)

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["mfcc"]).astype(np.float32)
    y = np.array(data["label"])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y).astype(np.int32)
    return X, y

def prepare_datasets(test_size, validation_size):
    X, y = load_data(JSON_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model():
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=10,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    return model

def switch_case(label_index):
    switcher = {
        0: "classical",
        1: "blues",
        2: "country",
        3: "disco",
        4: "hiphop",
        5: "jazz",
        6: "metal",
        7: "pop",
        8: "reggae",
        9: "rock"
    }
    return switcher.get(label_index, "unknown")

def predict(model, X, y):
    prediction = model.predict(X[np.newaxis, ...])
    expected_label = switch_case(y)
    predicted_label = switch_case(prediction[0])
    print("Expected label: {}, Predicted label: {}".format(expected_label, predicted_label))

if __name__ == "__main__":
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.15, 0.15)
    model = build_model()
    model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)], early_stopping_rounds=10, verbose=True)
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy on test set is: {}".format(test_accuracy))
    X = X_test[100]
    y = y_test[100]
    predict(model, X, y)
