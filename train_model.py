import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

# 1. Load the Delhi Traffic Dataset 
print("Step 1: Loading data...")
df = pd.read_csv('delhi_traffic_features.csv')

# --- RENAMING & CLEANING ---
df.columns = [col.lower().replace(' ', '_') for col in df.columns]
# Drop trip_id if it exists to avoid noise in the model
if 'trip_id' in df.columns:
    df_ml = df.drop(columns=['trip_id'])
else:
    df_ml = df.copy()

# 2. Preprocessing - One-Hot Encoding
print("Step 2: Encoding categorical variables...")
categorical_cols = ['start_area', 'end_area', 'time_of_day', 'day_of_week', 
                    'weather_condition', 'traffic_density_level', 'road_type']

# Ensure we only encode columns that actually exist in the CSV
existing_cats = [col for col in categorical_cols if col in df_ml.columns]
df_ml = pd.get_dummies(df_ml, columns=existing_cats)

# Define Features (X) and Target (y)
X = df_ml.drop(columns=['average_speed_kmph'])
y = df_ml['average_speed_kmph']

print(f"Dataset shape: {X.shape[0]} rows and {X.shape[1]} features.")

# 3. Train/Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training 
# Added n_jobs=-1 to use all CPU cores (Faster!)
print("Step 3: Training the Random Forest (this may take a minute)...")
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_test)
accuracy_r2 = r2_score(y_test, y_pred)

print("\n" + "="*30)
print(f"MODEL ACCURACY (R^2 Score): {accuracy_r2:.4f}")
print(f"Interpretation: The AI explains {accuracy_r2*100:.2f}% of traffic speed variance.")
print("="*30 + "\n")

# 6. Feature Importance Visualization
print("Step 4: Generating Importance Plot...")
importances = model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

# Take only top 10 for clarity
top_10 = feat_imp_df.head(10).copy()
top_10['Clean_Label'] = top_10['Feature'].apply(lambda x: x.split('_')[-1] if '_' in x else x)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_10, x='Importance', y='Clean_Label', hue='Clean_Label', palette='magma', legend=False)
plt.title(f'ATFOS Phase 1: Top 10 Traffic Factors (R^2: {accuracy_r2:.2f})')
plt.xlabel('Importance Score')
plt.ylabel('Traffic Features')
plt.tight_layout()

# Save plot and model
plt.savefig('feature_importance.png')
joblib.dump(model, 'atfos_model.pkl')
joblib.dump(X.columns.tolist(), 'atfos_features.pkl')

print("Phase 1 Complete: Model and plot saved successfully.")