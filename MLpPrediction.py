import os
from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd
# import tensorflow as tf
from keras import layers, models, Sequential
from keras.metrics import F1Score, Accuracy, Precision, Recall
from keras.src.saving import load_model
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.utils import to_categorical
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import Stocks

def evaluate_classification(y_true, y_pred):
    # Convert probabilities to class predictions
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Determine if binary or multi-class
    num_classes = len(np.unique(y_true))
    average_type = 'binary' if num_classes == 2 else 'macro'

    # Use sklearn for metrics
    accuracy = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels, average=average_type)
    recall = recall_score(y_true, y_pred_labels, average=average_type)
    f1 = f1_score(y_true, y_pred_labels, average=average_type)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }
def create_model_mlp(input_dim, first_layer, second_layer, output):
    model_mlp= models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(first_layer, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(second_layer, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(output, activation="softmax")
    ])


    return model_mlp

def evaluate_mlp_models(x_train, x_test, y_train, y_test):
    # Determine number of classes dynamically
    num_classes = len(np.unique(y_train))
    model = create_model_mlp(x_train.shape[1],32, 16, num_classes)



    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # print(model.summary())

    history = model.fit(
        x_train, y_train,
        epochs=30,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    y_pred = model.predict(x_test)
    return evaluate_classification(y_test, y_pred)

def get_train_test_data_for_MLP(ticker):


    # Load dataset
    df_clf = Stocks.get_data_stock(ticker, MLP=True)
    df_clf=df_clf.iloc[1:]



    # Impute numerical features if needed
    imputer = SimpleImputer(strategy='mean')
    X_clf = pd.DataFrame(imputer.fit_transform(df_clf), columns=df_clf.columns)


    # Split features and target

    y_clf = Stocks.create_data_output(X_clf)


    # Standardize the features
    scaler = StandardScaler()
    X_clf_scaled = scaler.fit_transform(X_clf)

    # Train-test split
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf_scaled, y_clf, test_size=0.2,
                                                                            random_state=42)

    # print("Classification data prepared. Shape:", X_train_clf.shape)
    return X_train_clf, X_test_clf, y_train_clf, y_test_clf


def find_best_model(ticker, f1_border=0.6):
    try:
        x_train, x_test, y_train, y_test = get_train_test_data_for_MLP(ticker)
    except Exception as e:
        print(f"Data error for {ticker}: {e}")
        return 0, None
    f1 = 0
    f1_best = -1
    best_model = None
    num_classes = len(np.unique(y_train))
    try_counter=0
    while f1<f1_border and try_counter<5:
        try_counter+=1
        layer1_units = random.randint(10, 64)
        layer2_units = random.randint(8, layer1_units)  # smaller than or equal to layer 1

        model = create_model_mlp(x_train.shape[1], layer1_units, layer2_units, num_classes)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        history = model.fit(
            x_train, y_train,
            epochs=30,
            batch_size=64,
            validation_split=0.2,
            verbose=1
        )

        y_pred = model.predict(x_test)
        # Convert probabilities to class predictions
        y_pred_labels = np.argmax(y_pred, axis=1)

        # Determine if binary or multi-class
        num_classes = len(np.unique(y_test))
        average_type = 'binary' if num_classes == 2 else 'macro'

        f1 = f1_score(y_test, y_pred_labels, average=average_type)
        if f1 > f1_best:
            f1_best = f1
            best_model = model

        # print(f"Score {f1}  model n:{try_counter}")
    return f1 , best_model



def get_or_create_model(ticker, f1_border=0.6):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
    MODEL_META_PATH = os.path.join(MODEL_DIR, "model_data.json")

    if os.path.exists(MODEL_META_PATH):
        model_data = pd.read_json(MODEL_META_PATH, orient="index")
    else:
        model_data = pd.DataFrame(columns=["filename", "date", "f1_score"])

    today = datetime.today()

    if ticker in model_data.index:
        last_trained = pd.to_datetime(model_data.loc[ticker, "date"])
        model_filename = model_data.loc[ticker, "filename"]
        model_path = os.path.join(MODEL_DIR, model_filename)

        # If model is fresh and file exists, load and return it
        if today - last_trained < timedelta(days=30) and os.path.exists(model_path):
            return load_model(model_path)

    # Otherwise: retrain
    try:
        f1, model = find_best_model(ticker, f1_border=f1_border)
        if model:
            model_filename = f"model_for{ticker}.keras"
            model_path = os.path.join(MODEL_DIR, model_filename)
            model.save(model_path)

            # Update metadata
            model_data.loc[ticker] = [model_filename, today.isoformat(), f1]
            model_data.to_json(MODEL_META_PATH, orient="index", date_format="iso")
            return model
    except Exception as e:
        return None


