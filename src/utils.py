from sklearn.metrics import classification_report, confusion_matrix

def print_evaluation(y_true, y_pred, title="Model Evaluation"):
    print(f"\n📋 {title}")
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))