from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def ex_svm(X_train, y_train, X_test, y_test, kernel='rbf', C=1):
    model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)

    print('Training SVM...')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f'Accuracy:  {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1 Score:  {f1:.4f}')

    return {
        'model': 'svm',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def ex_random_forest(X_train, y_train, X_test, y_test, n_estimators=100):
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
    print(classification_report(y_test, y_pred))

    return {
        'model': 'rf',
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
    }


def run_model(model_name, X_train, y_train, X_test, y_test):
    if model_name == 'svm':
        return ex_svm(X_train, y_train, X_test, y_test)
    if model_name == 'rf':
        return ex_random_forest(X_train, y_train, X_test, y_test)
    raise ValueError(f'Unknown model: {model_name}')
