import os

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ("log_cpc", "log_cpm")
FEATURE_WEIGHTS = np.array([0.6, 0.8])
TRAINING_SOURCES = (
    ("./cpc.csv", "./cpm.csv"),
    ("./data/cpc.csv", "./data/cpm.csv"),
)
RESULT_DIR = "./results"
ARTIFACTS = {
    "scaler": os.path.join(RESULT_DIR, "scaler.pkl"),
    "pca": os.path.join(RESULT_DIR, "pca.pkl"),
    "model": os.path.join(RESULT_DIR, "model.pkl"),
}
KMEANS_OPTIONS = {
    "n_clusters": 2,
    "init": "k-means++",
    "n_init": 20,
    "max_iter": 500,
    "random_state": 0,
}


def _ordered_log_features(records):
    records["timestamp"] = pd.to_datetime(records["timestamp"])
    records.sort_values(by="timestamp", inplace=True)
    records.reset_index(drop=True, inplace=True)

    records["log_cpc"] = np.log1p(records["cpc"])
    records["log_cpm"] = np.log1p(records["cpm"])
    return records.loc[:, FEATURE_COLUMNS]


def build_feature_frame(df):
    return _ordered_log_features(df)


def _first_available_pair(path_pairs):
    for cpc_path, cpm_path in path_pairs:
        if os.path.exists(cpc_path) and os.path.exists(cpm_path):
            return cpc_path, cpm_path
    raise FileNotFoundError("cpc.csv / cpm.csv not found in ./ or ./data/")


def read_training_data():
    cpc_path, cpm_path = _first_available_pair(TRAINING_SOURCES)
    cpc_data = pd.read_csv(cpc_path)
    cpm_data = pd.read_csv(cpm_path)
    print("training on", cpc_path, "+", cpm_path)
    return pd.merge(cpc_data, cpm_data, on="timestamp")


def _weighted_scaler(feature_matrix):
    normalizer = StandardScaler().fit(feature_matrix)
    # Fold feature weights into StandardScaler:
    # (x - mean) / (std / weight) == ((x - mean) / std) * weight.
    normalizer.scale_ = normalizer.scale_ / FEATURE_WEIGHTS
    return normalizer


def safe_dump(obj, path):
    if os.path.exists(path):
        os.remove(path)
    joblib.dump(obj, path)


def _save_artifacts(scaler, model):
    os.makedirs(RESULT_DIR, exist_ok=True)
    safe_dump(scaler, ARTIFACTS["scaler"])
    safe_dump(None, ARTIFACTS["pca"])
    safe_dump(model, ARTIFACTS["model"])


def train_policy():
    merged_data = read_training_data()
    feature_matrix = build_feature_frame(merged_data).values

    scaler = _weighted_scaler(feature_matrix)
    train_matrix = scaler.transform(feature_matrix)

    clusterer = KMeans(**KMEANS_OPTIONS)
    clusterer.fit(train_matrix)

    _save_artifacts(scaler, clusterer)

    print("saved policy to ./results")
    print("features =", list(FEATURE_COLUMNS), "with weights", FEATURE_WEIGHTS.tolist())
    print("kmeans = KMeans(n_clusters=2, n_init=20, max_iter=500, random_state=0)")


if __name__ == "__main__":
    train_policy()
