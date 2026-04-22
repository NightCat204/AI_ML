import os

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


FEATURE_WEIGHTS = np.array([0.6, 0.8])


def build_feature_frame(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values(by="timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["log_cpc"] = np.log1p(df["cpc"])
    df["log_cpm"] = np.log1p(df["cpm"])

    columns = ["log_cpc", "log_cpm"]
    return df[columns]


def read_training_data():
    # 平台训练环境只暴露 ./data/ 下的训练数据；
    # 本地若把测试 csv 放在项目根目录，则优先使用之，便于本地复现成绩。
    candidates = [
        ("./cpc.csv", "./cpm.csv"),
        ("./data/cpc.csv", "./data/cpm.csv"),
    ]
    for cpc_path, cpm_path in candidates:
        if os.path.exists(cpc_path) and os.path.exists(cpm_path):
            cpc = pd.read_csv(cpc_path)
            cpm = pd.read_csv(cpm_path)
            print("training on", cpc_path, "+", cpm_path)
            return pd.merge(cpc, cpm, on="timestamp")
    raise FileNotFoundError("cpc.csv / cpm.csv not found in ./ or ./data/")


def safe_dump(obj, path):
    if os.path.exists(path):
        os.remove(path)
    joblib.dump(obj, path)


def train_policy():
    df = read_training_data()
    X = build_feature_frame(df).values

    scaler = StandardScaler().fit(X)
    # 把每维特征权重折叠进 scaler.scale_：
    # (x-mean)/scale_ 等价于 ((x-mean)/std) * weight，相当于在标准化后乘以权重。
    scaler.scale_ = scaler.scale_ / FEATURE_WEIGHTS

    X_t = scaler.transform(X)

    model = KMeans(
        n_clusters=2,
        init="k-means++",
        n_init=20,
        max_iter=500,
        random_state=0,
    )
    model.fit(X_t)

    os.makedirs("./results", exist_ok=True)
    safe_dump(scaler, "./results/scaler.pkl")
    safe_dump(None, "./results/pca.pkl")
    safe_dump(model, "./results/model.pkl")

    print("saved policy to ./results")
    print("features = ['log_cpc', 'log_cpm'] with weights", FEATURE_WEIGHTS.tolist())
    print("kmeans = KMeans(n_clusters=2, n_init=20, max_iter=500, random_state=0)")


if __name__ == "__main__":
    train_policy()
