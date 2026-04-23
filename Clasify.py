import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def convert_categories_to_numbers(train_dataset, test_dataset):
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
    
    print(f"Rezultate pentru {name}:")
    print(f"Acuratete: {acuracy:.4f}")
    print(f"Precizie: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\n")
    
    return predictions

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
    processed_train_dataset, processed_test_dataset, label_encoder = convert_categories_to_numbers(train_dataset.copy(), test_dataset.copy())

    X_train = processed_train_dataset.drop(columns=['final_result', 'final_coursework_score'], errors='ignore')
    y_train = processed_train_dataset['final_result']
    
    X_test = processed_test_dataset.drop(columns=['final_result', 'final_coursework_score'], errors='ignore')
    y_test = processed_test_dataset['final_result']
    
    #good parameters for RandomForestClassifier
    pred = train_and_evaluate(X_train, X_test, y_train, y_test, None, 4, "RandomForestClassifier")
    plot_cm(y_test, pred, label_encoder, "RandomForestClassifier")
    # good parameters for DecisionTreeClassifier
    # pred2 = train_and_evaluate(X_train, X_test, y_train, y_test, 8, 20, "DecisionTreeClassifier")
    # plot_cm(y_test, pred2, label_encoder, "DecisionTreeClassifier")
    
