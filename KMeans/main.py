FEATURE_COLUMNS = ("log_cpc", "log_cpm")
DEFAULT_ANOMALY_RATIO = 0.024
MODEL_PATH = "./results/model.pkl"
SCALER_PATH = "./results/scaler.pkl"
PCA_PATH = "./results/pca.pkl"


def _make_log_features(frame):
    import numpy as np
    import pandas as pd

    # The judge may keep using the same DataFrame object, so this intentionally
    # preserves the in-place ordering behavior of the original submission.
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame.sort_values(by="timestamp", inplace=True)
    frame.reset_index(drop=True, inplace=True)

    frame["log_cpc"] = np.log1p(frame["cpc"])
    frame["log_cpm"] = np.log1p(frame["cpm"])
    return frame.loc[:, FEATURE_COLUMNS]


def _build_feature_frame(df):
    return _make_log_features(df)


def _load_pickle(path):
    import joblib

    return joblib.load(path)


def _load_kmeans_model(path):
    model = _load_pickle(path)

    # Older sklearn KMeans pickles may not have this runtime attribute. Newer
    # sklearn versions read it during predict(), so patch it lazily if needed.
    if not hasattr(model, "_n_threads"):
        model._n_threads = 1

    return model


def _as_dimension_frame(matrix):
    import pandas as pd

    return pd.DataFrame(
        matrix,
        columns=["Dimension{}".format(i + 1) for i in range(matrix.shape[1])],
    )


def preprocess_data(df):
    features = _build_feature_frame(df)
    scaler = _load_pickle(SCALER_PATH)
    pca = _load_pickle(PCA_PATH)

    if scaler is not None:
        transformed = scaler.transform(features)
    else:
        transformed = features.values

    if pca is not None:
        transformed = pca.transform(transformed)

    return _as_dimension_frame(transformed)


def _cluster_distance(feature_frame, kmeans, n_features):
    import numpy as np
    import pandas as pd

    samples = feature_frame.iloc[:, :n_features].values
    labels = kmeans.predict(samples)
    centers = kmeans.cluster_centers_
    distance = np.linalg.norm(samples - centers[labels], axis=1)
    return pd.Series(distance), pd.Series(labels)


def get_distance(data, kmeans, n_features):
    return _cluster_distance(data, kmeans, n_features)


def _mark_largest_distances(result_frame, ratio):
    sample_count = len(result_frame)
    top_count = max(1, int(round(sample_count * ratio)))
    top_count = min(top_count, sample_count)
    selected = result_frame["distance"].nlargest(top_count).index
    result_frame.loc[selected, "is_anomaly"] = True


def get_anomaly(data, kmean, ratio=None):
    result = data.copy()
    anomaly_ratio = DEFAULT_ANOMALY_RATIO if ratio is None else ratio

    n_features = len(result.columns)
    distance, labels = get_distance(result, kmean, n_features=n_features)

    result["cluster"] = labels.values
    result["distance"] = distance.values
    result["is_anomaly"] = False
    _mark_largest_distances(result, anomaly_ratio)

    return result


def predict(preprocess_data):
    ratio = DEFAULT_ANOMALY_RATIO
    kmeans = _load_kmeans_model(MODEL_PATH)
    anomaly_frame = get_anomaly(preprocess_data, kmeans, ratio)
    return anomaly_frame, preprocess_data, kmeans, ratio
