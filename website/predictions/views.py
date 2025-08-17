from pathlib import Path

import joblib
from django.conf import settings
from django.views.generic.edit import FormView

from .forms import PredictionForm


ML_MODELS_DIR = settings.BASE_DIR / "predictions" / "ml_models"

models = {}

scaler = joblib.load(ML_MODELS_DIR / "scaler.joblib")

for model_dir in ML_MODELS_DIR.iterdir():
    if model_dir.is_dir():
        model_file = model_dir / "model.joblib"
        if model_file.exists():
            models[model_dir.name] = joblib.load(model_file)


# Create your views here.
class PredictionFormView(FormView):
    template_name = "prediction_form.html"
    form_class = PredictionForm

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
        features_scaled = scaler.transform([features])

        high_risk_votes = 0

        for model_name, model in models.items():
            probability = model.predict_proba(features_scaled)[0][1]
            prediction = round(round(probability, 4) * 100, 4)
            results[model_name] = prediction

            if prediction >= 50:
                high_risk_votes += 1

        total_models = len(models)
        if high_risk_votes > (total_models / 2) - 1:
            message = (
                "The majority of the models conclude that you might have a risk of hypertension. "
                "We encourage you to get checked."
            )
        else:
            message = (
                "Majority of our models say that you don’t have a risk, but we still encourage you "
                "to get checked if you're unsure about your health."
            )

        return self.render_to_response(
            self.get_context_data(form=form, results=results, message=message)
        )
