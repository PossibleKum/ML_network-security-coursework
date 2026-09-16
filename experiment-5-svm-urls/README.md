# Experiment 3: SVM Classification of URLs

## Aim
Apply a Support Vector Machine (SVM) to classify URLs into legitimate and malicious categories.

## Objective
Develop a machine learning model that learns patterns from URL strings and predicts whether an unseen URL is legitimate or potentially malicious.

## Dataset
`urls.csv` contains a small synthetic educational dataset.

- `url` - URL string
- `label` - `0` for legitimate, `1` for malicious

## Method
1. Extract URL characteristics.
2. Represent URL patterns using character-level TF-IDF.
3. Train a linear SVM classifier.
4. Split data into training and testing sets.
5. Evaluate accuracy, precision, recall and F1-score.
6. Generate a confusion matrix.

## Run
From this folder:

```powershell
python url_svm.py
```

## Output
The program creates:

```text
confusion_matrix.png
classification_report.txt
```

## Note
The included dataset is synthetic and intended for laboratory demonstration. Its results should not be treated as performance of a production malicious-URL detector.
