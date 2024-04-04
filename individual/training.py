import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

label_encoder = LabelEncoder()
sc = StandardScaler()

# loading the dataset
data = pd.read_csv("individual/individual_carbonprint.csv")

# processing

# removing null values - people without vehicles
data.replace(np.nan, "None", inplace=True)
data.describe()

# take care of missing values
data.isna().sum()

# take care of categorical data
data.dtypes
data.nunique()

categorical_columns = data.select_dtypes(include=["object"]).columns

for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# training model
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# scaling the data
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

linearregression = LinearRegression()
decisiontreeregression = DecisionTreeRegressor()
supportvectorregression = SVR(kernel="rbf")
randomforestregression = RandomForestRegressor()
xgbregression = XGBRegressor()

linearregression.fit(X_train, y_train)
decisiontreeregression.fit(X_train, y_train)
supportvectorregression.fit(X_train, y_train)
randomforestregression.fit(X_train, y_train)
xgbregression.fit(X_train, y_train)

y_lin = linearregression.predict(X_test)
y_dectree = decisiontreeregression.predict(X_test)
y_supvec = supportvectorregression.predict(X_test)
y_randfor = randomforestregression.predict(X_test)
y_xgb = xgbregression.predict(X_test)

data1 = {
    "Regression Algorithms": [
        "Linear Regression",
        "Decision Tree Regression",
        "Support Vector Regression",
        "Random Forest Classifier",
        "XGB Regression",
    ],
    "Score": [
        r2_score(y_test, y_lin),
        r2_score(y_test, y_dectree),
        r2_score(y_test, y_supvec),
        r2_score(y_test, y_randfor),
        r2_score(y_test, y_xgb),
    ],
}

score = pd.DataFrame(data1)
print("Accuracy:")
print(score)

data2 = {
    "Regression Algorithms": [
        "Linear Regression",
        "Decision Tree Regression",
        "Support Vector Regression",
        "Random Forest Classifier",
        "XGB Regression",
    ],
    "Score": [
        mean_absolute_error(y_test, y_lin),
        mean_absolute_error(y_test, y_dectree),
        mean_absolute_error(y_test, y_supvec),
        mean_absolute_error(y_test, y_randfor),
        mean_absolute_error(y_test, y_xgb),
    ],
}


score2 = pd.DataFrame(data2)
print("mean absolute error")
print(score2)

# user_input = {
#     "Body Type": "overweight",
#     "Sex": "male",
#     "Diet": "pescatarian",
#     "How Often Shower": "daily",
#     "Heating Energy Source": "coal",
#     "Transport": "public",
#     "Vehicle Type": "None",
#     "Social Activity": "often",
#     "Monthly Grocery Bill": 230,
#     "Frequency of Traveling by Air": "frequently",
#     "Vehicle Monthly Distance Km": 210,
#     "Waste Bag Size": "large",
#     "Waste Bag Weekly Count": 4,
#     "How Long TV PC Daily Hour": 7,
#     "How Many New Clothes Monthly": 26,
#     "How Long Internet Daily Hour": 1,
#     "Energy efficiency": "No",
#     "Recycling": "['Metal']",
#     "Cooking_With": "['Stove', 'Oven']",
# }

# user_df = pd.DataFrame(user_input, index=[0])

# # Encode categorical variables using the label encoder
# for column in categorical_columns:
#     user_df[column] = label_encoder.fit_transform(user_df[column])

# # Scale the numerical features
# user_df_scaled = sc.transform(user_df)

# # Step 2: Predict Carbon Footprint

# # Predict carbon emissions using the trained Random Forest Regression model
# predicted_emissions_rf = randomforestregression.predict(user_df_scaled)
# predicted_emissions_xgb = xgbregression.predict(user_df_scaled)

# # Display the predicted carbon emissions to the user
# print("Predicted Carbon Emissions (Random Forest):", predicted_emissions_rf[0])
# print("Predicted Carbon Emissions (XGBoost):", predicted_emissions_xgb[0])


# # Save XGBRegressor model
# with open("xgboost_model.pkl", "wb") as file:
#     pickle.dump(xgbregression, file)

# # Save RandomForestRegressor model
# with open("random_forest_model.pkl", "wb") as file:
#     pickle.dump(randomforestregression, file)
