import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65

# Ty le lop duong tham chieu cua bo du lieu Adult (Bonus 5)
REFERENCE_POSITIVE_RATE = 0.248
POSITIVE_RATE_TOLERANCE = 0.05

# Mac dinh ghi vao file cuc bo; co the ghi de bang bien moi truong MLFLOW_TRACKING_URI.
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: canh bao lech phan phoi lop truoc khi huan luyen
    positive_rate = float(y_train.mean())
    if abs(positive_rate - REFERENCE_POSITIVE_RATE) > POSITIVE_RATE_TOLERANCE:
        print(
            f"::warning::LECH DU LIEU: ty le lop duong {positive_rate:.1%} lech qua "
            f"{POSITIVE_RATE_TOLERANCE * 100:.0f} diem phan tram so voi tham chieu "
            f"{REFERENCE_POSITIVE_RATE:.1%}"
        )
    else:
        print(f"Ty le lop duong {positive_rate:.1%} (tham chieu {REFERENCE_POSITIVE_RATE:.1%}): OK")

    with mlflow.start_run():
        mlflow.log_params(params)

        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # f1_score mac dinh tinh cho lop duong (target = 1); khong truyen average.
        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))

        # Bonus 2: quet nguong quyet dinh tu 0.1 den 0.9 (buoc 0.05).
        # f1_score o tren (nguong mac dinh 0.5) van la chi so de quality gate doc.
        proba = model.predict_proba(X_eval)[:, 1]
        thresholds = np.round(np.arange(0.10, 0.90 + 1e-9, 0.05), 2)
        f1_by_threshold = {
            float(t): float(f1_score(y_eval, (proba >= t).astype(int))) for t in thresholds
        }
        best_threshold = max(f1_by_threshold, key=f1_by_threshold.get)
        best_threshold_f1 = f1_by_threshold[best_threshold]

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("positive_rate", positive_rate)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_threshold_f1", best_threshold_f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")
        print(
            f"Nguong tot nhat: {best_threshold:.2f} -> F1 {best_threshold_f1:.4f} "
            f"(nguong 0.50 -> F1 {f1:.4f})"
        )

        # File nay duoc doc boi GitHub Actions o Buoc 2
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w") as f:
            json.dump(
                {
                    "f1_score": f1,
                    "accuracy": acc,
                    "positive_rate": positive_rate,
                    "best_threshold": best_threshold,
                    "best_threshold_f1": best_threshold_f1,
                    "f1_by_threshold": f1_by_threshold,
                },
                f,
                indent=2,
            )

        # File nay duoc upload len cloud storage o Buoc 2
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
