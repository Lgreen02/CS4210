import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data in chunks and concatenate them into a single DataFrame
data = pd.read_csv('open-meteo-one.csv')

# Split the concatenated DataFrame into 80% train and 20% test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Print to verify
print("Training data shape:", train_data.shape)
print("Test data shape:", test_data.shape)