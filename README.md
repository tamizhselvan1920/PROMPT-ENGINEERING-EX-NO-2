# EX-02-Cross-Platform-Prompting-Evaluating-Diverse-Techniques-in-AI-Powered-Text-Summarization

## AIM
To evaluate and compare the effectiveness of prompting techniques (zero-shot, few-shot, chain-of-thought, role-based) across different AI platforms (e.g., ChatGPT, Gemini, Claude, Copilot) in a specific task: text summarization.

## SCENARIO:
You are part of a content curation team for an educational platform that delivers quick summaries of research papers to undergraduate students. Your task is to summarize a 500-word technical article on "The Basics of Blockchain Technology" using multiple AI platforms and prompting strategies.

Your goal is to determine which combination of prompting technique + platform provides the best summary in terms of:
1. Accuracy
2. Coherence
3. Simplicity
4. Speed
5. User experience

## OUTPUT
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('D:/titanic_dataset.csv')
data.head()

# Data Cleansing - Replace Null Values
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = data[column].fillna(data[column].mode()[0])
    else:
        data[column] = data[column].fillna(data[column].mean())
print("Missing values handled.\n")

# Boxplot to Analyze Outliers (Fare)
plt.figure(figsize=(6,4))
sns.boxplot(x=data['Fare'])
plt.title("Boxplot - Fare")
plt.xlabel("Fare")
plt.show()

# Remove Outliers Using IQR Method
# Apply IQR method on 'Fare'
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]
data = remove_outliers_iqr(data, 'Fare')
print("Outliers removed using IQR method.\n")

# Countplot for Categorical Data
plt.figure(figsize=(7,4))
sns.countplot(x='Embarked', data=data)
plt.title("Countplot - Embarked Distribution")
plt.xticks(rotation=45)
plt.show()

# Displot for Univariate Distribution (Age)
sns.displot(data['Age'], kde=True, height=4, aspect=1.5)
plt.title("Displot - Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Cross Tabulation
crosstab_result = pd.crosstab(data['Sex'], data['Embarked'])
print("\nCross Tabulation Result (Sex vs Embarked):\n")
print(crosstab_result)

# Heatmap to Show Correlation
plt.figure(figsize=(8,6))
correlation_matrix = data.select_dtypes(include=np.number).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
```
<img width="603" height="480" alt="image" src="https://github.com/user-attachments/assets/f6587c0a-4572-42d8-8378-83fb19a96297" />

<img width="760" height="480" alt="image" src="https://github.com/user-attachments/assets/565b5f80-d159-445a-b6c0-0f15c2826184" />

<img width="737" height="503" alt="image" src="https://github.com/user-attachments/assets/331e40bd-5657-4f65-9404-56f2e9d53e4e" />

<img width="884" height="738" alt="image" src="https://github.com/user-attachments/assets/c87031ef-71d6-4d92-8390-6f3a12d1b91a" />

## RESULT:
Thus, evaluate and compare the effectiveness of prompting techniques (zero-shot, few-shot, chain-of-thought, role-based) across different AI platforms (e.g., ChatGPT, Gemini, Claude, Copilot) in a specific task: text summarization has been successfully done.
