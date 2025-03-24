<<<<<<< HEAD
# Description: This is a simple linear regression model that predicts the target variable based on a single feature.

print("Hello World 1")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 100 data points, single feature
y = 2.5 * X + np.random.randn(100, 1) * 2  # Linear relationship with some noise

# Convert to DataFrame for better visualization
df = pd.DataFrame(np.hstack((X, y)), columns=["Feature", "Target"])
print(df.head())
print ("hellow world 2")

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Visualize results
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.scatter(X_test, y_pred, color="red", label="Predicted")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.show()

=======
# Description: This is a simple linear regression model that predicts the target variable based on a single feature.

print("Hello World 1")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 100 data points, single feature
y = 2.5 * X + np.random.randn(100, 1) * 2  # Linear relationship with some noise

# Convert to DataFrame for better visualization
df = pd.DataFrame(np.hstack((X, y)), columns=["Feature", "Target"])
print(df.head())
print ("hellow world 2")

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Visualize results
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.scatter(X_test, y_pred, color="red", label="Predicted")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.show()

>>>>>>> c65f78c0bffddaa3ec1f96f82374451d3bc3d0cf
print ("hello world 3")