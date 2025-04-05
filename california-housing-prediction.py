# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the California Housing dataset
california = fetch_california_housing()
x = pd.DataFrame(california.data,columns=california.feature_names)
y = pd.DataFrame(california.target, columns=['MedHouseVal'])  # Median house value for California districts

# Display dataset information
print("California Housing Dataset Features:")
print(X.head())

# Display dataset information
print("\nTarget Variable (Median House Value):")
print(y.head())

# Split dataset
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=42)

# Train model
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\nModel Performance:")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Plot predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Diagonal line
plt.xlabel('Actual Prices ($1000s)')
plt.ylabel('Predicted Prices ($1000s)')
plt.title('Actual vs Predicted House Prices')
plt.grid(True)
plt.show()

# Display model coefficients (feature importance)
coefficients = pd.DataFrame({
    'Feature': california.feature_names,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("\nFeature Coefficients:")
print(coefficients)
