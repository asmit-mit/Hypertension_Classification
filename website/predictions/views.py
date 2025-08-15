from pathlib import Path

import joblib
from django.conf import settings
from django.shortcuts import render
from django.views.generic.edit import FormView

from .forms import PredictionForm

ML_MODELS_DIR = settings.BASE_DIR / "predictions" / "ml_models"

models = {}

for model_dir in ML_MODELS_DIR.iterdir():
    if model_dir.is_dir():
        model_file = model_dir / "model.joblib"
        if model_file.exists():
            models[model_dir.name] = joblib.load(model_file)


# Create your views here.
class PredictionFormView(FormView):
    template_name = "prediction_form.html"
    form_class = PredictionForm
    success_url = "/results/"

    def form_valid(self, form):
        features = [
            form.cleaned_data["age"],
            form.cleaned_data["salt_intake"],
            form.cleaned_data["stress_score"],
            form.cleaned_data["sleep_duration"],
            form.cleaned_data["bmi"],
            1 if form.cleaned_data["family_history"] == "Yes" else 0,
            1 if form.cleaned_data["exercise_level"] == "Low" else 0,
            1 if form.cleaned_data["exercise_level"] == "Moderate" else 0,
            1 if form.cleaned_data["smoking_status"] == "Smoker" else 0,
        ]

        results = {}
        for model_name, model in models.items():
            prediction = model.predict([features])[0]
            results[model_name] = prediction

        self.request.session["prediction_result"] = results

        return super().form_valid(form)
