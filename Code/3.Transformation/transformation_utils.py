from functools import reduce
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sn
import math
import copy
import sys
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def get_overall_age(birth_dates):
    birth_years = []
    for i in birth_dates:
        if pd.notna(i):  # Controlla se il valore non è null
            birth_years.append(int(i.split("-")[0]))
    return sum(birth_years) / len(birth_years) if birth_years else 0


def grid_search_features(df, excluded_cols, target, classifier, year, param_grid=None):
    num_features = len(df.drop(excluded_cols, axis=1).columns)
    if param_grid is None:
        param_grid = [
            (num_select, num_components)
            for num_select in range(
                math.floor(0.1 * num_features), math.ceil(0.9 * num_features)
            )
            for num_components in range(
                math.floor(0.1 * num_select), math.ceil(0.9 * num_select)
            )
        ]

    results = []

    for num_select, num_components in param_grid:
        new_classifier = copy.deepcopy(classifier)
        new_df = copy.deepcopy(df)

        new_df = select_features(new_df, target, excluded_cols, num_select)
        new_df = feature_aggregation_pca(new_df, num_components, excluded_cols)

        train_model_simple(new_classifier, new_df, year, target)

        (
            y_test,
            y_test_prob,
            conf_test,
            _,
            _,
            _,
            _
        ) = test_model(new_classifier, new_df, year, target)
        y_test_pred = enforce_max_teams(y_test_prob, conf_test)

        results.append(
            (
                num_select,
                num_components,
                round(accuracy_score(y_test, y_test_pred), 4) * 100,
                round(f1_score(y_test, y_test_pred), 4) * 100,
                round(roc_auc_score(y_test, y_test_prob), 4) * 100,
            )
        )

    results.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
    return results



def display_num_features_results(results):
    print(f"Accuracy \t F1 \t AUC \t SelectKBest \t PCA")
    for result in results:
        print(
            f"{result[2]} \t {result[3]} \t {result[4]} \t {result[0]} \t {result[1]}"
        )



def train_model_simple(classifier, df, year, target):
    x_train, y_train, _, _ = split_data(df, year, target)
    x_train = x_train.drop(["tmID"], axis=1)
    df["sampleWeight"] = df["year"].apply(
        lambda year_x: 2 ** (year - year_x - 1) if year > year_x else 1
    )
    try:
        classifier.fit(
            x_train, y_train, sample_weight=df.loc[x_train.index]["sampleWeight"]
        )
    except:
        classifier.fit(x_train, y_train)
    finally:
        df.drop("sampleWeight", axis=1, inplace=True)



def split_data(df, year, target):
    train_data = df[df["year"] < year]
    test_data = df[df["year"] == year]

    x_train = train_data.drop([target], axis=1)
    y_train = train_data[target]

    x_test = test_data.drop([target], axis=1)
    y_test = test_data[target]

    return x_train, y_train, x_test, y_test



def test_model(model, df, year, target):
    # Use to simultaneously test the model on the test and train data
    x_train, y_train, x_test, y_test = split_data(df, year, target)
    x_test_id = x_test["tmID"]

    x_train = x_train.drop(["tmID"], axis=1)
    x_test = x_test.drop(["tmID"], axis=1)

    y_test_prob = model.predict_proba(x_test)[:, 1]
    y_train_prob = model.predict_proba(x_train)[:, 1]

    return (
        y_test,
        y_test_prob,
        x_test["confID"],
        y_train,
        y_train_prob,
        x_train["confID"],
        x_test_id,
    )


def enforce_max_teams(y_prob, conf_id, max_teams=4):
    joined = zip(range(len(y_prob)), y_prob, conf_id)
    joined = sorted(joined, key=lambda x: x[1], reverse=True)

    y_pred = [0 for _ in range(len(y_prob))]

    count_0 = 0
    count_1 = 0
    for i, _, conf in joined:
        if count_0 < max_teams and conf == 0:
            y_pred[i] = 1
            count_0 += 1
        elif count_1 < max_teams and conf == 1:
            y_pred[i] = 1
            count_1 += 1
        else:
            continue

    return y_pred


def select_features(df, target, key_features, num_features=26):
    available_columns = list(set(df.columns) - set(key_features) - set(target))
    features_values = pd.DataFrame(df[available_columns], columns=available_columns)
    target_values = df["playoff"].values
    chi2(features_values, target_values)
    best_chi2_cols = SelectKBest(chi2, k=num_features)
    best_chi2_cols.fit(features_values, target_values)
    best_chi2_features = features_values.columns[best_chi2_cols.get_support()]

    key_predictors = set(best_chi2_features)
    key_predictors.update(key_features)

    df = df[list(key_predictors)]
    return df


def feature_aggregation_pca(df, n_components, columns_to_keep):
    column_names = [
        f"PC{i + 1}" for i in range(n_components)
    ]  # Create custom column names

    df_to_keep = pd.DataFrame(df[columns_to_keep])
    df.drop(columns_to_keep, axis=1, inplace=True)

    # Assuming 'X' is your data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df)

    pca = PCA(n_components=n_components)  # Choose the number of components you want
    x_pca = pca.fit_transform(x_scaled)

    df_result = pd.DataFrame(data=x_pca, columns=column_names)

    for col in columns_to_keep:
        df_result[col] = df_to_keep[col].reset_index(drop=True)

    return df_result