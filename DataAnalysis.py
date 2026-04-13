import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

def analyze_numeric_attributes(dataset, numeric_atribute_columns):
    stats = dataset[numeric_atribute_columns].describe().T #calculate statistics
    stats['non_null_count'] = dataset[numeric_atribute_columns].notna().sum() #count non-null values

    print("Numeric Attributes Statistics:")
    print(stats[['non_null_count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']])
    print("\n")

    for col in numeric_atribute_columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=dataset[col])
        plt.title(f'{col}')
        plt.savefig(f'boxplot_{col}.png')
        plt.close()

def analyze_categorical_attributes(dataset, categorical_atribute_columns):
    stats = pd.DataFrame({
        'non_null_count': dataset[categorical_atribute_columns].notna().sum(), #count non-null values
        'unique_values': dataset[categorical_atribute_columns].nunique() #count unique values
    })

    print("Categorical Attributes Statistics:")
    print(stats)
    print("\n")

    for col in categorical_atribute_columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(y=dataset[col], order=dataset[col].value_counts().index)
        plt.tight_layout()
        plt.savefig(f'countplot_{col}.png')
        plt.close()

def analyze_class_balance(dataset, target_col):
    plt.figure(figsize=(8, 4))
    sns.countplot(x=dataset[target_col], order=dataset[target_col].value_counts().index)
    plt.tight_layout()
    plt.savefig('class_balance.png')
    plt.close()

def analyze_all_correlations(dataset, num_col, cat_col, class_target, reg_target):
    correlation = dataset[[num_col, reg_target]].corr().iloc[0, 1]
    print(f"Pearson correlation between: {num_col} and {reg_target}: {correlation:.4f}")
    plt.figure(figsize=(8, 4))
    sns.scatterplot(x=dataset[num_col], y=dataset[reg_target], alpha=0.5)
    plt.title(f'{num_col} vs {reg_target}')
    plt.savefig(f'scatter_{num_col}_vs_{reg_target}.png')
    plt.close()

    crosstab_result = pd.crosstab(index=dataset[cat_col], columns=dataset[class_target])
    chi2_result = chi2_contingency(crosstab_result)
    print(f"\nChi-Square test: {cat_col} vs {class_target}:")
    print(crosstab_result)
    print(f"P-Value: {chi2_result[1]:.4f}\n")

    plt.figure(figsize=(8, 4))
    sns.boxplot(x=dataset[class_target], y=dataset[num_col])
    plt.title(f'{num_col} on classes of {class_target}')
    plt.savefig(f'boxplot_{num_col}_vs_{class_target}.png')
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.boxplot(x=dataset[cat_col], y=dataset[reg_target])
    plt.title(f'{reg_target} on categories of {cat_col}')
    plt.xticks(rotation=45)
    plt.savefig(f'boxplot_{cat_col}_vs_{reg_target}.png', bbox_inches='tight')
    plt.close()