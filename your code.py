import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load and Enhance Dataset
df = pd.read_csv("fraud dataset .csv", encoding='ISO-8859-1')

# Simulate additional transaction data
np.random.seed(42)
df['TransactionAmount'] = np.random.randint(50, 10000, size=len(df))
df['Merchant'] = np.random.choice(['ELECTRONICS', 'GROCERY', 'TRAVEL', 'RESTAURANT', 'CLOTHING'], size=len(df))
df['Location'] = np.random.choice(['NEW YORK', 'LOS ANGELES', 'CHICAGO', 'HOUSTON', 'MIAMI'], size=len(df))

# Step 2: Encode Categorical Data
le_profession = LabelEncoder()
le_merchant = LabelEncoder()
le_location = LabelEncoder()

df['Profession'] = le_profession.fit_transform(df['Profession'])
df['Merchant'] = le_merchant.fit_transform(df['Merchant'])
df['Location'] = le_location.fit_transform(df['Location'])

# Step 3: Train the Model
features = ['Profession', 'Income', 'TransactionAmount', 'Merchant', 'Location']
X = df[features]
y = df['Fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 4: Interactive Prediction System
def check_transaction():
    print("\n---- Credit Card Fraud Detection System ----")

    profession_input = input(f"Enter Profession {le_profession.classes_.tolist()}: ").strip().upper()
    income = int(input("Enter Annual Income: "))
    amount = int(input("Enter Transaction Amount: "))
    merchant_input = input(f"Enter Merchant Type {le_merchant.classes_.tolist()}: ").strip().upper()
    location_input = input(f"Enter Transaction Location {le_location.classes_.tolist()}: ").strip().upper()

    # Validation
    if profession_input not in le_profession.classes_:
        print("Invalid profession.")
        return
    if merchant_input not in le_merchant.classes_:
        print("Invalid merchant type.")
        return
    if location_input not in le_location.classes_:
        print("Invalid location.")
        return

    # Encode inputs
    prof_encoded = le_profession.transform([profession_input])[0]
    merch_encoded = le_merchant.transform([merchant_input])[0]
    loc_encoded = le_location.transform([location_input])[0]

    user_data = [[prof_encoded, income, amount, merch_encoded, loc_encoded]]
    prediction = model.predict(user_data)[0]

    print("\n>>> Prediction Result <<<")
    if prediction == 1:
        print("ALERT: This transaction is likely FRAUDULENT.")
    else:
        print("This transaction appears to be LEGITIMATE.")

# Run the checker
check_transaction()