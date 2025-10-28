from django.urls import include, path

from . import views

urlpatterns = [
    path("", views.PredictionFormView.as_view(), name="prediction_form"),
    path("send_mail", views.send_email, name="send_email"),
]
