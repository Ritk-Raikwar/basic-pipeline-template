import seaborn as sns
import os
import pandas as pd
data = sns.load_dataset("iris")

folder_path = "data"
file_name = "iris_dataset.csv"
full_path = os.path.join(folder_path, file_name)

os.makedirs(folder_path, exist_ok=True)

data.to_csv(full_path, index=False)
print(f"Data saved to: {full_path}")