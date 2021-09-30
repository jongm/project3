"""
Code for checking model performance on different slices
"""

import pandas as pd
from sklearn.model_selection import train_test_split

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

def check_slices_performance(data):

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=777)

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

    val_data["pred"] = predictions
    val_data["label"] = lb.transform(val_data["salary"])

    # Calculating slice results:
    slice_results = pd.DataFrame()
    for group in cat_features:

        performance = val_data.groupby(group).apply(
            lambda df: compute_model_metrics(df["label"], df["pred"], as_df=True)
        )
        performance = performance.droplevel(1)
        performance.index.name = "group_value"
        performance["group"] = group
        slice_results = slice_results.append(performance)

    slice_results = slice_results.reset_index()
    slice_results = slice_results[["group","group_value","precision","recall","fbeta","FPR"]]

    return slice_results


if __name__=="__main__":

    # Loading the data:
    data = pd.read_csv("census_clean.csv")

    output = check_slices_performance(data)

    # Printing results to file:
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_rows', None)
    with open("slice_output.txt",'w') as file:
        print(output.to_string(), file=file)

