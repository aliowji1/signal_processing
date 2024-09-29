import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import numpy as np


#Patient Averages and Age Plot

# Read the CSV data
data = pd.read_csv('spg_average_features.csv')

# Extract relevant columns
metric_1 = data['Metric1']
metric_2 = data['Metric2']
category_1 = data['Category1']
category_2 = data['Category2']

# Create lists for each category
group_1_a_metric1 = []
group_1_a_metric2 = []
group_1_b_metric1 = []
group_1_b_metric2 = []
group_2_a_metric1 = []
group_2_a_metric2 = []
group_2_b_metric1 = []
group_2_b_metric2 = []

# Categorize data
for i in range(len(data)):
    if category_1[i] == 'A' and category_2[i] == 'X':
        group_1_a_metric1.append(metric_1[i])
        group_1_a_metric2.append(metric_2[i])
    elif category_1[i] == 'B' and category_2[i] == 'X':
        group_1_b_metric1.append(metric_1[i])
        group_1_b_metric2.append(metric_2[i])
    elif category_1[i] == 'A' and category_2[i] == 'Y':
        group_2_a_metric1.append(metric_1[i])
        group_2_a_metric2.append(metric_2[i])
    elif category_1[i] == 'B' and category_2[i] == 'Y':
        group_2_b_metric1.append(metric_1[i])
        group_2_b_metric2.append(metric_2[i])

# Set a larger font size and bold font
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# Create the plot
plt.figure(figsize=(14, 8))  # Increased size for better visibility
plt.scatter(group_1_a_metric2, group_1_a_metric1, color='red', marker='o', s=200, label='A, X')
plt.scatter(group_1_b_metric2, group_1_b_metric1, color='red', marker='^', s=200, label='B, X')
plt.scatter(group_2_a_metric2, group_2_a_metric1, color='green', marker='o', s=200, label='A, Y')
plt.scatter(group_2_b_metric2, group_2_b_metric1, color='green', marker='^', s=200, label='B, Y')

plt.xlabel('Metric 2', fontsize=16)
plt.ylabel('Metric 1', fontsize=16)
plt.title('Metric 1 vs Metric 2', fontsize=18)

# Position legend to the left with larger font
plt.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0., fontsize=14)

plt.grid(True, linestyle='--', alpha=0.7)

# Make tick labels larger
plt.tick_params(axis='both', which='major', labelsize=14)

# Adjust layout to prevent cutoff
plt.tight_layout()

plt.show()


#Regressions

# Read the CSV file
df = pd.read_csv('new_average_features.csv')

# Filter for the specified subjects
subjects = [1, 3, 12, 14, 16, 22, 23, 24, 30, 31, 35, 36]
df_filtered = df[df['Subject'].isin([f'subject{i}' for i in subjects])]

# Get all feature columns (excluding 'Subject', 'Waveform Index', 'PrimaryMetric', 'Metric1', 'Metric2', 'Category')
feature_columns = df_filtered.columns.drop(['Subject', 'Waveform Index', 'PrimaryMetric', 'Metric1', 'Metric2', 'Category'])

# Filter out non-numeric columns
numeric_feature_columns = df_filtered[feature_columns].select_dtypes(include=[np.number]).columns

# Create a color map for categories
categories = df_filtered['Category'].unique()
color_map = plt.cm.get_cmap('Set1')(np.linspace(0, 1, len(categories)))
color_dict = dict(zip(categories, color_map))

# Create regression plots for each feature
for feature in numeric_feature_columns:
    plt.figure(figsize=(10, 6))
    
    x = df_filtered[feature]
    y = df_filtered['PrimaryMetric']
    
    # Remove any rows with NaN values
    mask = ~(x.isna() | y.isna())
    x = x[mask]
    y = y[mask]
    
    try:
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Plot the scatter points, color-coded by Category
        for category in categories:
            category_mask = df_filtered['Category'] == category
            category_mask = category_mask & mask  # Combine with non-NaN mask
            plt.scatter(x[category_mask], y[category_mask], c=[color_dict[category]], label=category, alpha=0.7)
        
        # Plot the regression line
        plt.plot(x, slope * x + intercept, color='black', linestyle='--', label=f'R² = {r_value**2:.4f}')
        
        plt.xlabel(feature)
        plt.ylabel('PrimaryMetric')
        plt.title(f'{feature} vs PrimaryMetric')
        plt.legend()
        plt.show()
        plt.close()
        print(f"Plot for {feature} created successfully")
    except Exception as e:
        print(f"Error creating plot for {feature}: {str(e)}")
        plt.close()

print("All regression plots have been created.")