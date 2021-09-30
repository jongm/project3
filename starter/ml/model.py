import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier(max_depth=10, random_state=777)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds, as_df=False):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    as_df : bool
        Returns the output as a single row dataframe
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)

    # Get false positive rate:
    FP = sum((preds == 1) & (y == 0))
    TN = sum((preds == 0) & (y == 0))
    FPR = FP / (FP + TN)

    if as_df is True:
        return pd.DataFrame({"precision":precision,
                             "recall":recall,
                             "fbeta":fbeta,
                             "FPR":FPR}, index=[0])
    else:
        return precision, recall, fbeta, FPR


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
