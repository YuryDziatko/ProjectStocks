import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import Stocks

def get_train_test_data():
    # Load dataset
    df_clf = Stocks.get_data_stock()

    # Check for missing values
    print("Missing values per column:")
    print(df_clf.isnull().sum())

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

    print("Classification data prepared. Shape:", X_train_clf.shape)
    return X_train_clf, X_test_clf, y_train_clf, y_test_clf

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_classification(y_true, y_pred):

    results = {
        "Accuracy":accuracy_score(y_true, y_pred),
        "Precision":precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred)
    }
    return results

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def evaluate_classification_models(x_train, x_test, y_train, y_test):
    models = {
        "LinearRegression": LogisticRegression(),
        "Ridge": DecisionTreeClassifier(),
        "Lasso": RandomForestClassifier(),
        "DecisionTree": SVC(),
        "RandomForest": KNeighborsClassifier()
    }

    results_reg = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred_model = model.predict(x_test)
        results_reg[name] = evaluate_classification(y_test, y_pred_model)

    return results_reg


X_train, X_test, Y_train, Y_test = get_train_test_data()
results = evaluate_classification_models(X_train, X_test, Y_train, Y_test)
for name, evaluate in results.items():
    print(f"Model: {name}")
    print(f"evaluate: {evaluate}")