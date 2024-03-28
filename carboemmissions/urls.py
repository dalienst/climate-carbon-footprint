# urls.py
from django.urls import path
from carboemmissions.views import CarbonEmissionPredictionView, CarbonEmissionPredictionModelsView

urlpatterns = [
    path(
        "predict-carbon-emissions/",
        CarbonEmissionPredictionView.as_view(),
        name="predict_carbon_emissions",
    ),
    path("predict/", CarbonEmissionPredictionModelsView.as_view(), name="predict-models")
]
