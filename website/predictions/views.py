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
        age = form.cleaned_data["age"]
        salt_intake = form.cleaned_data["salt_intake"]
        stress_score = form.cleaned_data["stress_score"]
        sleep_duration = form.cleaned_data["sleep_duration"]
        bmi = form.cleaned_data["bmi"]
        family_history = form.cleaned_data["family_history"]
        exercise_level = form.cleaned_data["exercise_level"]
        smoking_status = form.cleaned_data["smoking_status"]

        if bmi < 18.5:
            bmi_underweight = 1
            bmi_overweight = 0
            bmi_obese = 0
        elif 25 <= bmi < 30:
            bmi_underweight = 0
            bmi_overweight = 1
            bmi_obese = 0
        elif bmi >= 30:
            bmi_underweight = 0
            bmi_overweight = 0
            bmi_obese = 1
        else:
            bmi_underweight = 0
            bmi_overweight = 0
            bmi_obese = 0

        if age < 30:
            age_young = 1
            age_middle = 0
            age_senior = 0
        elif 30 <= age < 60:
            age_young = 0
            age_middle = 1
            age_senior = 0
        else:
            age_young = 0
            age_middle = 0
            age_senior = 1

        if salt_intake <= 8:
            salt_moderate = 0
            salt_high = 0
        elif 8 < salt_intake <= 10:
            salt_moderate = 1
            salt_high = 0
        else:
            salt_moderate = 0
            salt_high = 1

        risk_score = (
            (age / 100) * 0.3 +
            (bmi / 40) * 0.25 +
            (salt_intake / 15) * 0.2 +
            (stress_score / 10) * 0.15 +
            (1 if family_history == "Yes" else 0) * 0.1
        )

        features = [
            age,
            salt_intake,
            stress_score,
            sleep_duration,
            bmi,
            risk_score,
            1 if family_history == "Yes" else 0,
            1 if exercise_level == "Low" else 0,
            1 if exercise_level == "Moderate" else 0,
            1 if smoking_status == "Smoker" else 0,
            bmi_obese,
            bmi_overweight,
            bmi_underweight,
            age_middle,
            age_senior,
            age_young,
            salt_moderate,
            salt_high,
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
