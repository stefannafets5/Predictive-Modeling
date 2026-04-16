import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve

def plot_learning_curves(X_train, y_train, X_test, y_test, models, model_names):
    plt.figure(figsize=(12, 8))
    
    # generate 10 sets form 10% to 100% of data
    train_sizes_fractions = np.linspace(0.1, 1.0, 10)
    n_samples = len(X_train)

    for i in range(len(models)):
        model = models[i]
        name = model_names[i]
        
        train_errors = []
        val_errors = []
        
        for fraction in train_sizes_fractions:
            subset_size = int(fraction * n_samples)
            
            X_subset = X_train.iloc[:subset_size]
            y_subset = y_train.iloc[:subset_size]
            
            model.fit(X_subset, y_subset)
            
            # evaluare pe setul de antrenament curent
            train_preds = model.predict(X_subset)
            train_errors.append(mean_absolute_error(y_subset, train_preds))
            
            # evaluare pe setul de validare (test) complet
            val_preds = model.predict(X_test)
            val_errors.append(mean_absolute_error(y_test, val_preds))
            
        # adaugam liniile in grafic
        plt.plot(train_sizes_fractions * 100, train_errors, linestyle='--', label=f'{name} (Train)')
        plt.plot(train_sizes_fractions * 100, val_errors, marker='o', label=f'{name} (Validation)')

    plt.xlabel('Training dataset size')
    plt.ylabel('Mean Absolute Error')
    plt.title('Regression Model Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()

def run_regression(train_dataset, test_dataset):
    # prepare data
    X_train = train_dataset.drop(columns=['final_result', 'final_coursework_score'], errors='ignore')
    y_train = train_dataset['final_coursework_score']
    
    X_test = test_dataset.drop(columns=['final_result', 'final_coursework_score'], errors='ignore')
    y_test = test_dataset['final_coursework_score']
    
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    model_lr = LinearRegression()
    model_ridge = Ridge(alpha=5.0)
    model_rf = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=20)

    models = [model_lr, model_ridge, model_rf]
    names = ['LinearRegression', 'RidgeRegression', 'RandomForestRegression']

    # evaluate each model
    for i in range(len(models)):
        model = models[i]
        name = names[i]
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        print(f"result for {name}:")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print("\n")

    plot_learning_curves(X_train, y_train, X_test, y_test, models, names)