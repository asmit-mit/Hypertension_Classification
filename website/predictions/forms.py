from django import forms


class PredictionForm(forms.Form):
    age = forms.IntegerField(label="Age")
    salt_intake = forms.FloatField(label="Salt Intake (grams/day)")
    stress_score = forms.FloatField(label="Stress Score (0-10)")
    sleep_duration = forms.FloatField(label="Sleep Duration (hours/day)")
    bmi = forms.FloatField(label="Body Mass Index (BMI)")

    family_history = forms.ChoiceField(
        choices=[("Yes", "Yes"), ("No", "No")], label="Family History of Hypertension"
    )

    exercise_level = forms.ChoiceField(
        choices=[("Low", "Low"), ("Moderate", "Moderate"), ("High", "High")], label="Exercise Level"
    )

    smoking_status = forms.ChoiceField(
        choices=[("Smoker", "Smoker"), ("Non-Smoker", "Non-Smoker")], label="Smoking Status"
    )
