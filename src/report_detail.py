"""Bonus 3: bao cao precision / recall theo tung lop va confusion matrix dang van ban.

Doc models/model.joblib va data/holdout.csv, ghi ket qua ra outputs/detail.txt.
"""
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

CLASS_NAMES = ["thu_nhap_thap (0)", "thu_nhap_cao (1)"]


def build_report(model_path="models/model.joblib", eval_path="data/holdout.csv") -> str:
    model = joblib.load(model_path)
    df = pd.read_csv(eval_path)
    X, y = df.drop(columns=["target"]), df["target"]
    preds = model.predict(X)

    cm = confusion_matrix(y, preds, labels=[0, 1])
    precision, recall, f1, support = precision_recall_fscore_support(
        y, preds, labels=[0, 1], zero_division=0
    )

    lines = [
        "CONFUSION MATRIX (hang = thuc te, cot = du doan)",
        f"{'':<20}{'du doan thap':>14}{'du doan cao':>14}",
        f"{'thuc te thap':<20}{cm[0, 0]:>14}{cm[0, 1]:>14}",
        f"{'thuc te cao':<20}{cm[1, 0]:>14}{cm[1, 1]:>14}",
        "",
        "PRECISION / RECALL THEO LOP",
        f"{'lop':<20}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}",
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(
            f"{name:<20}{precision[i]:>10.4f}{recall[i]:>10.4f}{f1[i]:>10.4f}{support[i]:>10}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import os

    text = build_report()
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/detail.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
