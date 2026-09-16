import pandas as pd
import re

# Load URL dataset
df = pd.read_csv("experiment-6/urls.csv")


# Extract useful features from each URL
def extract_features(url):
    return pd.Series({
        "length": len(url),
        "dots": url.count("."),
        "slashes": url.count("/"),
        "digits": sum(c.isdigit() for c in url),
        "special_chars": len(re.findall(r"[^a-zA-Z0-9]", url)),
        "has_ip": int(bool(re.search(r"\d+\.\d+\.\d+\.\d+", url))),
        "suspicious_word": int(any(
            word in url.lower()
            for word in ["login", "verify", "secure", "update", "password"]
        ))
    })


# Apply feature engineering
features = df["url"].apply(extract_features)

# Combine original URLs with extracted features
result = pd.concat([df, features], axis=1)

# Detect abnormal patterns
abnormal = result[
    (result["length"] > 75) |
    (result["special_chars"] > 15) |
    (result["digits"] > 5) |
    (result["has_ip"] == 1) |
    (result["suspicious_word"] == 1)
]

print("=" * 60)
print("URL FEATURE ENGINEERING - ABNORMAL PATTERN DETECTION")
print("=" * 60)

print("\nAll URL Features:")
print(result)

print("\n\nAbnormal / Suspicious URL Patterns:")
print(abnormal)

# Save results
abnormal.to_csv("abnormal_urls.csv", index=False)

print("\nResults saved to abnormal_urls.csv")