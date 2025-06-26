import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.metrics import F1Score, Accuracy, Precision, Recall
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.utils import to_categorical
from keras.datasets import cifar10

from ModelPrediction import get_train_test_data

# def evaluate_classification(y_true, y_pred):
#
#
#     results = {
#         "Accuracy":Accuracy(y_true, y_pred),
#         "Precision":Precision(y_true, y_pred),
#         "Recall": Recall(y_true, y_pred),
#         "F1-score": F1Score(y_true, y_pred)
#     }
#     return results
#
# def evaluate_mlp_models(x_train, x_test, y_train, y_test):
#     model_mlp = models.Sequential([
#         layers.Input(shape=x_train[0].shape),
#         layers.Dense(100, activation="relu"),
#         layers.Dense(len(np.unique(y_train)), activation="softmax")
#     ])
#
#     model_mlp.summary()
#     model_mlp.compile(
#         optimizer="adam",
#         loss="sparse_categorical_crossentropy",
#         metrics=["arruracy"])
#
#     model_mlp.fit(x_train, y_train, epochs=10, batch_size=100)
#     y_pred_model = model_mlp.predict(x_test)
#     results_mlp = evaluate_classification(y_test, y_pred_model)
#     return results_mlp
#
#
#
#
# X_train, X_test, Y_train, Y_test = get_train_test_data()
# results = evaluate_mlp_models(X_train, X_test, Y_train, Y_test)
# print(results)


# def evaluate_classification(y_true, y_pred):
#     # Convert probabilities to class predictions
#     y_pred_labels = np.argmax(y_pred, axis=1)
#
#     # Handle binary vs multi-class classification
#     num_classes = len(np.unique(y_true))
#     average_type = 'binary' if num_classes == 2 else 'macro'
#
#     # Initialize metrics
#     accuracy = Accuracy()(y_true, y_pred_labels)
#     precision = Precision(average=average_type)(y_true, y_pred_labels)
#     recall = Recall(average=average_type)(y_true, y_pred_labels)
#     f1 = F1Score(average=average_type, name='f1_score')(y_true, y_pred_labels)
#
#     return {
#         "Accuracy": accuracy.numpy(),
#         "Precision": precision.numpy(),
#         "Recall": recall.numpy(),
#         "F1-score": f1.numpy()
#     }
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


def evaluate_mlp_models(x_train, x_test, y_train, y_test):
    # Determine number of classes dynamically
    num_classes = len(np.unique(y_train))

    model = models.Sequential([
        layers.Input(shape=(x_train.shape[1],)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(model.summary())

    history = model.fit(
        x_train, y_train,
        epochs=30,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    y_pred = model.predict(x_test)
    return evaluate_classification(y_test, y_pred)


# Main execution
if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = get_train_test_data()
    results = evaluate_mlp_models(X_train, X_test, Y_train, Y_test)

    print("\nFinal Evaluation Metrics:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")