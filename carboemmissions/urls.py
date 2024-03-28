# urls.py
from django.urls import path
from carboemmissions.views import CarbonEmissionPredictionView

urlpatterns = [
    path(
        "predict-carbon-emissions/",
        CarbonEmissionPredictionView.as_view(),
        name="predict_carbon_emissions",
    ),
]
