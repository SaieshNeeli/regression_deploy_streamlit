import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

# Step 1: Prepare Data (Area vs. Price)
data = {
    "area": [500, 700, 800, 1000, 1500, 1800, 2000, 2500, 3000, 3500],
    "price": [100, 150, 170, 210, 290, 320, 350, 420, 500, 600]
}

df = pd.DataFrame(data)

# Step 2: Split into Features (X) and Target (y)
X = df[['area']]  # Feature (area)
y = df['price']   # Target (price)

# Step 3: Train the Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# Step 4: Save the Trained Model
with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained and saved as 'house_price_model.pkl'")
