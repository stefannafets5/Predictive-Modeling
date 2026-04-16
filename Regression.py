import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_regression(train_dataset, test_dataset):
    # prepare data
    X_train = train_dataset.drop(columns=['final_result', 'final_coursework_score'], errors='ignore')
    y_train = train_dataset['final_coursework_score']
    
    X_test = test_dataset.drop(columns=['final_result', 'final_coursework_score'], errors='ignore')
    y_test = test_dataset['final_coursework_score']
    
    # One-hot encoding
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # generate graph
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valori Reale')
    plt.ylabel('Predictii')
    plt.title('Regresie Liniara: Real vs Predictie')
    plt.savefig('regression_results.png')
    plt.close()