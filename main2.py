import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("student.csv")
print("First 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())
df = df.drop_duplicates()
df = df.fillna(df.mean(numeric_only=True))
corr = df.corr(numeric_only=True)
print("\nCorrelation Matrix:")
print(corr)
print(corr.shape)
df.hist(figsize=(10,8))
plt.suptitle("Histograms")
plt.show()
plt.figure(figsize =(8,6))
df.boxplot()
plt.title("Boxplot")
plt.show()
plt.figure(figsize=(8,6))
if not corr.empty: 
  sns.heatmap(corr,annot=True,cmap="coolwarm")
  plt.title("Correlation Heatmap")
  plt.show()
else:
   print("No numeric columns found for correlation.")
print("\nKey Insights:")
for col in corr.columns:
    strong_corr = corr[col][abs(corr[col]) > 0.7]
    if len(strong_corr) > 1:
        print(f"\n{col} has strong correlation with:")
        print(strong_corr)