def _build_feature_frame(df):
    import numpy as np
    import pandas as pd

    # 平台测评会复用传入对象，这里保持原地修改
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values(by="timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["log_cpc"] = np.log1p(df["cpc"])
    df["log_cpm"] = np.log1p(df["cpm"])

    columns = ["log_cpc", "log_cpm"]
    return df[columns]


def _load_kmeans_model(path):
    import joblib

    model = joblib.load(path)

    # 兼容旧版 sklearn 序列化出的 KMeans。
    # 在 1.6.x 环境下，旧模型可能缺少 _n_threads，调用 predict 会直接报错。
    if not hasattr(model, "_n_threads"):
        model._n_threads = 1

    return model


def preprocess_data(df):
    import pandas as pd
    import joblib

    data = _build_feature_frame(df)

    scaler = joblib.load("./results/scaler.pkl")
    pca = joblib.load("./results/pca.pkl")

    if scaler is not None:
        data = scaler.transform(data)
    else:
        data = data.values

    if pca is not None:
        data = pca.transform(data)

    data = pd.DataFrame(
        data,
        columns=["Dimension{}".format(i + 1) for i in range(data.shape[1])],
    )
    return data


def get_distance(data, kmeans, n_features):
    import numpy as np
    import pandas as pd

    X = data.iloc[:, :n_features].values
    labels = kmeans.predict(X)
    centers = kmeans.cluster_centers_

    distance = np.linalg.norm(X - centers[labels], axis=1)
    return pd.Series(distance), pd.Series(labels)


def get_anomaly(data, kmean, ratio=None):
    data = data.copy()

    if ratio is None:
        ratio = 0.024

    n_features = len(data.columns)
    distance, labels = get_distance(data, kmean, n_features=n_features)

    data["cluster"] = labels.values
    data["distance"] = distance.values
    data["is_anomaly"] = False

    top_k = max(1, int(round(len(data) * ratio)))
    top_k = min(top_k, len(data))
    anomaly_index = data["distance"].nlargest(top_k).index
    data.loc[anomaly_index, "is_anomaly"] = True

    return data


def predict(preprocess_data):
    ratio = 0.024

    kmeans = _load_kmeans_model("./results/model.pkl")
    is_anomaly = get_anomaly(preprocess_data, kmeans, ratio)

    return is_anomaly, preprocess_data, kmeans, ratio
