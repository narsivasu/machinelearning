from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
X = np.array([[1000], [1500], [1800], [2400], [3000]])
y = np.array([300, 400, 450, 580, 700])
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pre=model.predict(X_test)

print(f"Slope (Weight):    {model.coef_[0]:.2f}")
print(f"Intercept (Bias):  {model.intercept_:.2f}")
print(f"Mean Squared Error:{mean_squared_error(y_test, y_pre):.2f}")
print(f"R² Score:          {r2_score(y_test, y_pre):.2f}")
