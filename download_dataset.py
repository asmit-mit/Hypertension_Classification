import kagglehub

# Download latest version
path = kagglehub.dataset_download("miadul/hypertension-risk-prediction-dataset")

print("Path to dataset files:", path)
