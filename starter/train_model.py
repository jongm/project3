"""
Code for training the classification model
"""

import pickle

import pandas as pd
from sklearn.model_selection import KFold

from starter.ml.model import train_model, compute_model_metrics, inference
from starter.ml.data import process_data

# Categorical features:
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]

# =============================================================================
# K-Fold cross validation
# =============================================================================

def kfold_validation(data):

    kf = KFold(n_splits=5, shuffle=True, random_state=777)

    kfold_results = []

    for train_index, test_index in kf.split(data):

        # Splitting data:
        train_data = data.iloc[train_index]
        val_data = data.iloc[test_index]

        # Processing training data:
        X_train, y_train, encoder, lb, scaler = process_data(
            train_data, categorical_features=cat_features, label="salary", training=True
        )

        # Processing validation data:
        X_val, y_val, encoder, lb, scaler = process_data(
            val_data, categorical_features=cat_features, label="salary", training=False,
            encoder=encoder, lb=lb, scaler=scaler
        )

        # Creating and fitting model:
        model = train_model(X_train, y_train)

        # Measuring model performance:
        predictions = inference(model, X_val)
        precision, recall, fbeta = compute_model_metrics(y_val, predictions)
        kfold_results.append({"precision":precision,
                              "recall":recall,
                              "fbeta":fbeta,})

    # Summary of cross validation:
    kfold_results = pd.DataFrame(kfold_results).mean()
    print("Model performance after Kfolds:")
    print(kfold_results)

    return kfold_results


# =============================================================================
# Final model training:
# =============================================================================

def train_final_model(data):

    # Processing whole data:
    X, y, encoder, lb, scaler = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )

    # Creating and fitting model:
    model = train_model(X, y)

    # Saving model and encoders:
    with open("./models/inference_model.pkl", "wb") as file:
        pickle.dump(model, file)

    with open("./models/onehot_encoder.pkl", "wb") as file:
        pickle.dump(encoder, file)

    with open("./models/label_encoder.pkl", "wb") as file:
        pickle.dump(lb, file)

    with open("./models/scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)


if __name__=="__main__":

    # Loading the data:
    data = pd.read_csv("census_clean.csv")

    train_final_model(data)
