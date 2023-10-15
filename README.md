import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Load the dataset from CSV
df = pd.read_csv('Housing.csv')
df
# Summary statistics of the dataset
print(df.describe())
# Check for missing values
print(df.isnull().sum())
# Correlation matrix to understand feature relationships
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
# Preprocessing: Selecting features and target variable
X = df[['bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'parking']]
y = df['price']
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)
# Create a Linear Regression model
model = LinearRegression()
