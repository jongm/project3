import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler


def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None, scaler=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
        Can be a dict, then it will be transformed to a DF
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    # If given a dict, converts to pd DataFrame:
    if isinstance(X, dict):
        X = pd.DataFrame(X, index=[0])

    # Changes column names from "_" to "-" since pydantic BaseModel does
    # not allow "-" in field names:
    X.columns = [col.replace("_", "-") for col in X.columns]

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()

        scaler = StandardScaler()
        X_continuous = scaler.fit_transform(X_continuous)
    else:
        X_categorical = encoder.transform(X_categorical)
        X_continuous = scaler.transform(X_continuous)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            y = None

    X = np.concatenate([X_continuous, X_categorical], axis=1)

    return X, y, encoder, lb, scaler