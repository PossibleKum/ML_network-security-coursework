from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "urls.csv"
CM_FILE = BASE_DIR / "confusion_matrix.png"
REPORT_FILE = BASE_DIR / "classification_report.txt"


def extract_url_features(url: str) -> str:
    url = str(url).strip().lower()

    features = [
        url,
        f"url_length_{len(url)}",
        f"dot_count_{url.count('.')}",
        f"slash_count_{url.count('/')}",
        f"hyphen_count_{url.count('-')}",
        f"underscore_count_{url.count('_')}",
        f"question_count_{url.count('?')}",
        f"equal_count_{url.count('=')}",
        f"ampersand_count_{url.count('&')}",
        f"at_count_{url.count('@')}",
        f"colon_count_{url.count(':')}",
        f"digit_count_{sum(ch.isdigit() for ch in url)}",
        f"subdomain_count_{max(url.count('.') - 1, 0)}",
        f"has_https_{int(url.startswith('https://'))}",
        f"has_ip_{int(bool(re.search(r'https?://(?:\d{1,3}\.){3}\d{1,3}', url)))}",
        f"has_exe_{int('.exe' in url)}",
        f"has_php_{int('.php' in url)}",
        f"has_suspicious_word_{int(any(
            word in url
            for word in ['verify', 'login', 'secure', 'update',
                         'confirm', 'free', 'urgent', 'password']
        ))}",
    ]
    return " ".join(features)


def main():
    df = pd.read_csv(DATA_FILE)

    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'url' and 'label' columns.")

    df = df.dropna(subset=["url", "label"]).copy()
    df["label"] = df["label"].astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0 for legitimate and 1 for malicious.")

    df["features"] = df["url"].apply(extract_url_features)

    X_train, X_test, y_train, y_test = train_test_split(
        df["features"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=1,
            sublinear_tf=True,
        )),
        ("scaler", StandardScaler(with_mean=False)),
        ("svm", SVC(kernel="linear", C=1.0)),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=["Legitimate", "Malicious"],
        zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 60)
    print("SVM URL CLASSIFICATION")
    print("=" * 60)
    print(f"Dataset size: {len(df)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm)
    ax.set_title("SVM URL Classification - Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_xticks([0, 1], ["Legitimate", "Malicious"])
    ax.set_yticks([0, 1], ["Legitimate", "Malicious"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    plt.savefig(CM_FILE, dpi=150, bbox_inches="tight")
    plt.show()

    REPORT_FILE.write_text(
        f"SVM URL Classification Results\n{'=' * 32}\n"
        f"Dataset size: {len(df)}\n"
        f"Training samples: {len(X_train)}\n"
        f"Testing samples: {len(X_test)}\n"
        f"Accuracy: {accuracy:.4f}\n\n{report}",
        encoding="utf-8",
    )

    print(f"\nConfusion matrix saved to: {CM_FILE}")
    print(f"Classification report saved to: {REPORT_FILE}")
    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()
