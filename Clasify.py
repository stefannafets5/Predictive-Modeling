import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

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
    # model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    model = RandomForestClassifier(n_estimators=250, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
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

    # RandomForestClassifier test
    mprec = 0
    macc = 0
    mrec = 0
    mf1 = 0
    for i in range (10):
        pred, acc, prec, rec, f1 = train_and_evaluate(X_train, X_test, y_train, y_test, None, 4, "RandomForestClassifier")
        mprec += prec
        macc += acc
        mrec += rec
        mf1 += f1
    
    print("pt 4")
    print(f"Acuratete: {macc/10:.4f}")
    print(f"Precizie: {mprec/10:.4f}")
    print(f"Recall: {mrec/10:.4f}")
    print(f"F1 Score: {mf1/10:.4f}")
    print("\n")
    
    plot_cm(y_test, pred, label_encoder, "RandomForestClassifier")

    # DecisionTreeClassifier test

    # mprec = 0
    # macc = 0
    # mrec = 0
    # mf1 = 0
    # for i in range (10):
    #     pred2, acc, prec, rec, f1 = train_and_evaluate(X_train, X_test, y_train, y_test, 8, 20, "DecisionTreeClassifier")
    #     mprec += prec
    #     macc += acc
    #     mrec += rec
    #     mf1 += f1
    
    # print("pt decision")
    # print(f"Acuratete: {macc/10:.4f}")
    # print(f"Precizie: {mprec/10:.4f}")
    # print(f"Recall: {mrec/10:.4f}")
    # print(f"F1 Score: {mf1/10:.4f}")
    # print("\n")
    
    # plot_cm(y_test, pred2, label_encoder, "DecisionTreeClassifier")