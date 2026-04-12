import pandas as pd
from DataAnalysis import (
    analyze_numeric_attributes, 
    analyze_categorical_attributes, 
    analyze_class_balance,
    analyze_all_correlations
)

def analyze_dataset(dataset):
    # analyze columns
    numeric_features = ['studied_credits', 'mean_score_early']
    categorical_features = ['disability', 'highest_education']
    
    analyze_numeric_attributes(dataset, numeric_features)
    analyze_categorical_attributes(dataset, categorical_features)
    
    class_target = 'final_result'    
    analyze_class_balance(dataset, class_target)
    
    reg_target = 'final_coursework_score'
    analyze_all_correlations(dataset, 'studied_credits', 'highest_education', class_target, reg_target)

def main():
    # load data
    print("Incarcare dataset...")
    train_dataset = pd.read_csv('CA_OUALD_train.csv', skipinitialspace=True)
    
    analyze_dataset(train_dataset)

if __name__ == "__main__":
    main()