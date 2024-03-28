# views.py
import numpy as np
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


class CarbonEmissionPredictionView(APIView):
    def post(self, request):
        user_input = request.data

        # Load the dataset
        data = pd.read_csv("carboemmissions/individual_carbonprint.csv")

        # Preprocess data
        data.replace(np.nan, "None", inplace=True)
        categorical_columns = data.select_dtypes(include=["object"]).columns
        label_encoder = LabelEncoder()
        for column in categorical_columns:
            data[column] = label_encoder.fit_transform(data[column])

        # Extract features and target variable
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # Scale features
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X)

        # Train models
        xgbregression = XGBRegressor()
        randomforestregression = RandomForestRegressor()

        xgbregression.fit(X_scaled, y)
        randomforestregression.fit(X_scaled, y)

        # Preprocess user input
        user_df = pd.DataFrame(user_input, index=[0])
        for column in categorical_columns:
            user_df[column] = label_encoder.fit_transform(user_df[column])
        user_df_scaled = sc.transform(user_df)

        # Predict carbon emissions using the trained models
        predicted_emissions_rf = randomforestregression.predict(user_df_scaled)
        predicted_emissions_xgb = xgbregression.predict(user_df_scaled)

        # Calculate the average of predicted carbon emissions from both models
        predicted_emissions_avg = (predicted_emissions_rf + predicted_emissions_xgb) / 2

        # Return the predicted carbon emissions as a response, including the average
        return Response(
            {
                "predicted_carbon_emissions_rf": predicted_emissions_rf[0],
                "predicted_carbon_emissions_xgb": predicted_emissions_xgb[0],
                "predicted_carbon_emissions_avg": predicted_emissions_avg[0],
            },
            status=status.HTTP_200_OK,
        )
