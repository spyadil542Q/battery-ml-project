import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

df = pd.read_csv("battery_monitoring_project/data/Battery_dataset.csv") #df is the data frame (dataset loaded in pandas)
print(df.head()) #df.head shows first few row of of the data set. By default it shows first 5 row
print(df.shape) #df.shape returns the dimension of the data set..here it is 680 rows and 11 columns..680 battery observation and 11 parameters

plt.figure() #opens a new graph canvas
plt.plot(df["cycle"],df["SOH"])
plt.xlabel("Cycle number")
plt.ylabel("State of health (SOH)")
plt.title("Battery degradation over the cycles")
plt.show() #without this line plot will not appear in a py script
plt.figure()
plt.scatter(df["cycle"],df["chT"])
plt.xlabel("Cycle number")
plt.ylabel("Charging temp")
plt.title("Charging temp variation vs cycle number")
plt.show()
plt.figure()
plt.plot(df["cycle"], df["BCt"])
plt.xlabel("Cylce number")
plt.ylabel("Battery capacity")
plt.title("battery capacity vs cycle number")
plt.show()
plt.figure()
plt.scatter(df["cycle"], df["chV"])
plt.xlabel("Cycle number")
plt.ylabel("Charging voltage")
plt.title("Charging voltage vs cycle number")
plt.show()
plt.figure()
plt.scatter(df["cycle"], df["disV"])
plt.xlabel("Cycle number")
plt.ylabel("Discharging voltage")
plt.title("Discharging voltage vs cycle number")
plt.show()

print(df.corr(numeric_only=True))
X = df.drop(columns=["SOH","RUL","battery_id"])
y = df["SOH"]
print(X.head())
print(y.head())
plt.figure()

#outlier detection
df.boxplot(column=["chT","chV","disV","disT","chI","disI"])
plt.title("Outlier detection in battery parameters")
plt.show()

#model building starts: but first splitting the data
from sklearn.model_selection import  train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("Training samples:", X_train.shape)
print("testing samples:",X_test.shape)

#fitting the data and training
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

#predicting values from the model we just built and trained
y_pred = model.predict(X_test)

#calc the error to check deviation of the value we are getting from the actual value
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print("Mean squared error:", mse)
print("R2 score:", r2)
mae = mean_absolute_error(y_test, y_pred)

print("Linear Regression Results:")
print("MSE:", mse)
print("R2:", r2)
print("MAE:", mae)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print("\nRandom Forest Results:")
print("MSE:", mse_rf)
print("R2:", r2_rf)
print("MAE:", mae_rf)

# Feature Importance (Random Forest)
import pandas as pd

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance)
plt.figure()
plt.bar(feature_importance["Feature"], feature_importance["Importance"])
plt.xticks(rotation=45)
plt.title("Feature Importance (Random Forest)")
plt.show()
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual SOH")
plt.ylabel("Predicted SOH")
plt.title("Actual vs Predicted Battery SOH")
plt.show()

# Predict Remaining Useful Life (RUL) now
y_rul = df["RUL"]
X_rul = df.drop(columns=["SOH", "RUL", "battery_id"])

# split data
X_train_rul, X_test_rul, y_train_rul, y_test_rul = train_test_split(
    X_rul, y_rul, test_size=0.2, random_state=42
)

# create model
model_rul = LinearRegression()

# train
model_rul.fit(X_train_rul, y_train_rul)

# predictions
y_pred_rul = model_rul.predict(X_test_rul)
from sklearn.ensemble import RandomForestRegressor

rf_model_rul = RandomForestRegressor(random_state=42)
rf_model_rul.fit(X_train_rul, y_train_rul)

y_pred_rul_rf = rf_model_rul.predict(X_test_rul)

# evaluation
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Linear Regression metrics
mse_rul = mean_squared_error(y_test_rul, y_pred_rul)
r2_rul = r2_score(y_test_rul, y_pred_rul)
mae_rul = mean_absolute_error(y_test_rul, y_pred_rul)

print("Linear RUL Model:")
print("MSE:", mse_rul)
print("R2:", r2_rul)
print("MAE:", mae_rul)

# Random Forest metrics
mse_rul_rf = mean_squared_error(y_test_rul, y_pred_rul_rf)
r2_rul_rf = r2_score(y_test_rul, y_pred_rul_rf)
mae_rul_rf = mean_absolute_error(y_test_rul, y_pred_rul_rf)

print("\nRandom Forest RUL Model:")
print("MSE:", mse_rul_rf)
print("R2:", r2_rul_rf)
print("MAE:", mae_rul_rf)
plt.figure()
plt.scatter(y_test_rul, y_pred_rul)
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Actual vs Predicted Remaining Useful Life")
plt.show()
plt.figure()
plt.scatter(y_test_rul, y_pred_rul_rf)
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL (RF)")
plt.title("RF: Actual vs Predicted RUL")
plt.show()

def battery_health_status(soh):

    if soh > 90:
        return "Excellent"

    elif soh > 70:
        return "Good"

    elif soh > 50:
        return "Degraded"

    else:
        return "Replace Battery"
    
sample_soh = float(y_pred[0])
print("Predicted SOH:", sample_soh)
print("Battery Status:", battery_health_status(sample_soh))
def fault_detection(row):

    if row["chT"] > 35:
        return "Overheating Warning"

    if row["chV"] > 4.35:
        return "Overvoltage Warning"

    if row["disV"] < 2.8:
        return "Deep Discharge Warning"

    return "Normal"

print(fault_detection(df.sample().iloc[0]))