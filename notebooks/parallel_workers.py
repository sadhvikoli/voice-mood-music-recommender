from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



def _resolve_labels_and_names(y_train, y_test, class_names=None):
    if class_names is not None:
        labels = list(range(len(class_names)))
        names = [str(name) for name in class_names]
        return labels, names

    labels = sorted(np.unique(np.concatenate([y_train, y_test])).tolist())
    names = [str(label) for label in labels]
    return labels, names


def ex_svm(X_train, y_train, X_test, y_test, class_names=None, kernel='rbf', C=1):
    labels, target_names = _resolve_labels_and_names(y_train, y_test, class_names)
    model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)

    print('Training SVM...')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - SVM (with Parallel Workers)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_svm.png')

    print(f'Accuracy:  {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1 Score:  {f1:.4f}')
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0))

    return {
        'model': 'svm',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0),
    }


def ex_random_forest(X_train, y_train, X_test, y_test, class_names=None, n_estimators=100):
    labels, target_names = _resolve_labels_and_names(y_train, y_test, class_names)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
    )

    print('Training Random Forest...')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f'Accuracy:  {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall:    {rec:.4f}')
    print(f'F1-Score:  {f1:.4f}')

    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=target_names,
            yticklabels=target_names)
    plt.title('Confusion Matrix - Random Forest (with Parallel Workers)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_rf.png')
    # plt.show()

    return {
        'model': 'rf',
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'classification_report': classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0),
    }


def run_model(model_name, X_train, y_train, X_test, y_test, class_names=None):
    if model_name == 'svm':
        return ex_svm(X_train, y_train, X_test, y_test, class_names=class_names)
    if model_name == 'rf':
        return ex_random_forest(X_train, y_train, X_test, y_test, class_names=class_names)
    raise ValueError(f'Unknown model: {model_name}')
