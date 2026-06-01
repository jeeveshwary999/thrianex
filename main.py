import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("data.csv")
print("Original Data")
print(df)
df["Age"] = pd.to_numeric(df["Age"], errors='coerce')
df["Salary"] = pd.to_numeric(df["Salary"], errors='coerce')
df = df.drop_duplicates()
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())
df = df[df["Age"] <= 50]
print("\nCleaned Data")
print(df)
df.to_csv("cleaned_data.csv", index = False)
plt.figure(figsize = (6,4))
sns.histplot(df["Salary"],bins = 5)
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Count")
plt.show()