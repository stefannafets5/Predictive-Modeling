import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def encode_dataset(train_dataset, test_dataset):
    target = 'final_result'
    encoder = LabelEncoder()

    train_dataset[target] = encoder.fit_transform(train_dataset[target])
    train_dataset = pd.get_dummies(train_dataset)

    test_dataset[target] = encoder.transform(test_dataset[target])
    test_dataset = pd.get_dummies(test_dataset)

    # for columns that don't exist in the test dataset
    test_dataset = test_dataset.reindex(columns=train_dataset.columns, fill_value=0)

    return train_dataset, test_dataset, encoder

def train_and_evaluate(X_train, X_test, y_train, y_test, max_depth, min_samples_leaf, name):
    if(name == "RandomForestClassifier"):
        model = RandomForestClassifier(n_estimators=250, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    elif (name == "DecisionTreeClassifier"):
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    else:
        print("Invalid model name")
        return

    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    acuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    
    return predictions, acuracy, precision, recall, f1

def plot_cm(y_test, predictions, label_encoder, model_name):
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f'conf_matrix_{model_name}.png')
    plt.close()

def run_model(train_dataset, test_dataset):
    processed_train_dataset, processed_test_dataset, label_encoder = encode_dataset(train_dataset.copy(), test_dataset.copy())

    X_train = processed_train_dataset.drop(columns=['final_result', 'final_coursework_score'], errors='ignore')
    y_train = processed_train_dataset['final_result']
    
    X_test = processed_test_dataset.drop(columns=['final_result', 'final_coursework_score'], errors='ignore')
    y_test = processed_test_dataset['final_result']

    run_classifier(0, X_train, X_test, y_train, y_test, label_encoder)

def run_classifier(idx, X_train, X_test, y_train, y_test, label_encoder):
    mprec = 0
    macc = 0
    mrec = 0
    mf1 = 0
    bprec = 0
    bacc = 0
    brec = 0
    bf1 = 0
    for i in range (10):
        if (idx == 0):
            pred, acc, prec, rec, f1 = train_and_evaluate(X_train, X_test, y_train, y_test, None, 4, "RandomForestClassifier")
        else:
            pred, acc, prec, rec, f1 = train_and_evaluate(X_train, X_test, y_train, y_test, 8, 20, "DecisionTreeClassifier")
        mprec += prec
        macc += acc
        mrec += rec
        mf1 += f1
        if (bf1 < f1):
            bf1 = f1
            bprec = prec
            bacc = acc
            brec = rec
    
    print("pt 4")
    print(f"Medium accuracy: {macc/10:.4f} maximum: {bacc:.4f}")
    print(f"Medium precision: {mprec/10:.4f} maximum: {bprec:.4f}")
    print(f"Medium recall: {mrec/10:.4f} maximum: {brec:.4f}")
    print(f"Medium F1 Score: {mf1/10:.4f} maximum: {bf1:.4f}")
    print("\n")
    
    if (idx == 0):
        plot_cm(y_test, pred, label_encoder, "RandomForestClassifier")
    else:
        plot_cm(y_test, pred, label_encoder, "DecisionTreeClassifier")
