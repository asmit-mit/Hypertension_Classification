from django.urls import include, path

from . import views

urlpatterns = [
    path("", views.PredictionFormView.as_view(), name="prediction_form"),
]
