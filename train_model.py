import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv('data/veg_price.csv')

# Rename columns
df = df.rename(columns={
    'Arrival_Date': 'date',
    'Modal_x0020_Price': 'modal_price',
    'Min_x0020_Price': 'min_price',
    'Max_x0020_Price': 'max_price'
})

# Convert to datetime
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

# Extract date features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Select features and target
X = df[['year', 'month', 'day', 'State', 'District', 'Market', 'Commodity', 'Variety', 'Grade']]
y = df['modal_price']

# One-hot encode features
X_encoded = pd.get_dummies(X)

# Save training columns for use in Flask app
training_columns = X_encoded.columns
pd.DataFrame(training_columns).to_csv('training_columns.csv', index=False, header=False)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'veg_price_model.pkl')

print("Training complete, model and columns saved.")
