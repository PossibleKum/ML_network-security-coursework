import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
# ---------------- Step 1: Create Synthetic Dataset ----------------
data = {
"url": [
# Legitimate URLs
"https://www.google.com",
"https://www.amazon.com/login",
"https://github.com/openai",
"https://www.microsoft.com/en-us/",
"https://www.paypal.com/home",
"https://www.bankofamerica.com/security",
"https://www.linkedin.com/in/johndoe",
"https://www.apple.com/store",
# Phishing URLs
"http://paypal-login-security-alert.com",
"http://verify-account-amazon.biz/login",
"https://secure-facebook-login-update.ru",
"http://bankofamerica-login-confirm.net",
"https://google-account-reset-now.xyz",
"http://microsoft-support-verification.top",
"http://appleid-reset-alert.org",
"https://login-update-github.io"
],
"label": [0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1] # 0=Legit, 1=Phish
}
df = pd.DataFrame(data)
print("✅ Synthetic phishing URL dataset created.")
print(df.head())
# ---------------- Step 2: Data Preprocessing ----------------
X = df["url"]
y = df["label"]
# Convert text to numerical vectors using TF-IDF (NLP feature extraction)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,5))
# Character-level n-grams help catch patterns like 'login', 'verify', etc.
X_vectorized = vectorizer.fit_transform(X)
# ---------------- Step 3: Split Data ----------------
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y,
test_size=0.3, random_state=42)
# ---------------- Step 4: Train Model ----------------
model = LogisticRegression()
model.fit(X_train, y_train)
# ---------------- Step 5: Evaluate Model ----------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {acc*100:.2f}%")
print("\n✅ Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
xticklabels=['Legit','Phish'], yticklabels=['Legit','Phish'])
plt.title("Confusion Matrix - Phishing URL Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Legit', 'Phish']))
# ---------------- Step 6: Save Model & Vectorizer ----------------
joblib.dump(model, "phishing_url_model.pkl")
joblib.dump(vectorizer, "phishing_vectorizer.pkl")
print("\n✅ Model and vectorizer saved successfully.")
# ---------------- Step 7: Test with New URLs ----------------
new_urls = [
"https://secure-login-paypal-alert.info",
"https://www.amazon.in/gp/cart/view.html",
"http://update-facebook-verification.com",
"https://accounts.google.com"
]
new_features = vectorizer.transform(new_urls)
predictions = model.predict(new_features)
print("\n✅ URL Prediction Results:")
for url, pred in zip(new_urls, predictions):
    status = "✅ Phishing" if pred == 1 else "✅ Legitimate"
    print(f"{url} --> {status}")