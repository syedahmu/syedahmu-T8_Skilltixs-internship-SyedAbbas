# House Price Prediction with Feature Engineering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('train.csv')

# Drop columns with too many missing values
df.drop(columns=['Alley', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True)

# Handle missing values
df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)
df['MasVnrType'].fillna('None', inplace=True)
df['MasVnrArea'].fillna(0, inplace=True)
df['GarageType'].fillna('None', inplace=True)
df['GarageFinish'].fillna('None', inplace=True)
df['GarageQual'].fillna('None', inplace=True)
df['GarageCond'].fillna('None', inplace=True)
df['GarageYrBlt'].fillna(df['GarageYrBlt'].median(), inplace=True)
df['BsmtQual'].fillna('None', inplace=True)
df['BsmtCond'].fillna('None', inplace=True)
df['BsmtExposure'].fillna('None', inplace=True)
df['BsmtFinType1'].fillna('None', inplace=True)
df['BsmtFinType2'].fillna('None', inplace=True)
df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)
df['FireplaceQu'].fillna('None', inplace=True)

# Drop ID column
df.drop('Id', axis=1, inplace=True)

# Log transformation on target
df['SalePrice'] = np.log1p(df['SalePrice'])

# Log transform skewed numeric features
numeric_feats = df.select_dtypes(include=[np.number])
skewness = numeric_feats.skew().sort_values(ascending=False)
skewed_features = skewness[skewness > 1].index

df[skewed_features] = np.log1p(df[skewed_features])

# Label Encoding for ordinal features
ordinal_features = ['ExterQual', 'BsmtQual', 'KitchenQual']
le = LabelEncoder()
for col in ordinal_features:
    df[col] = le.fit_transform(df[col])

# One-hot encoding for categorical variables
df = pd.get_dummies(df)

# Split data
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(20)

# Plot top 20 features
plt.figure(figsize=(10, 6))
sns.barplot(x=top_features, y=top_features.index)
plt.title("Top 20 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
