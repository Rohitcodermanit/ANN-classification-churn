# ANN Classification — Customer Churn Prediction
A complete Artificial Neural Network (ANN) pipeline to predict customer churn—built with preprocessing, model training, evaluation, and prediction components.

## Project Overview
This project builds an ANN-based classifier to predict whether a bank customer will churn (i.e., stop using the service), based on their profile data. The workflow includes data preprocessing, model training, saving the trained model, and using it for predictions—all wrapped inside a Python scripting and notebook environment.

## Repository Structure
ANN-classification-churn/
├── Churn_Modelling.csv # Raw dataset with customer features
├── experiments.ipynb # Notebook for EDA, preprocessing, model training, evaluation
├── prediction.ipynb # Notebook demonstrating model inference
├── app.py # (Optional) Application script to use the model for predictions
├── model.h5 # Saved ANN model
├── label_encoder_gender.pkl # Encoder for gender feature
├── onehot_encoder_geo.pkl # Encoder for geography feature
├── scaler.pkl # Scaler for numerical feature normalization
├── requirements.txt # Python dependencies
└── README.md # This documentation

## Getting Started
### Prerequisites
- Python 3.x
- Jupyter Notebook (optional, for experimenting)
- Required Python libraries specified in `requirements.txt`

### Installation
cmd
git clone https://github.com/Rohitcodermanit/ANN-classification-churn.git
cd ANN-classification-churn
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Usage
Training & Experimentation
Open and run the experiments.ipynb notebook. It typically walks through the following steps:
Data Exploration — Analyze features, visualizations, class distributions.
Preprocessing — Encoding categorical variables (gender, geography), scaling numeric features, handling data splits.
Model Building — Defining ANN architecture, choosing activation and optimizer.
Model Training & Evaluation — Training the model and evaluating its performance with metrics like accuracy, precision, recall, etc.
Model Saving — Storing the trained model (model.h5) and relevant preprocessing objects (scaler.pkl, encoders).
Making Predictions
Using prediction.ipynb — Load the saved model and preprocessing objects to predict churn probability or label on new customer profiles.
Using app.py (if provided) — A Python script that loads the model and provides a simple interface (could be CLI or web) to run predictions programmatically.

Example usage in notebook or script:
import pickle
from tensorflow.keras.models import load_model

# Load encoders and model
le_gender = pickle.load(open("label_encoder_gender.pkl", "rb"))
ohe_geo = pickle.load(open("onehot_encoder_geo.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
model = load_model("model.h5")

# Prepare input data accordingly, encode, scale, and predict
Model Architecture & Performance
The project uses a feedforward Artificial Neural Network (ANN) for binary classification (churn vs. no churn).
Architecture likely includes input, hidden, and output layers built with Keras or TensorFlow (details available in experiments.ipynb).
Evaluation metrics—such as accuracy, confusion matrix, and possibly ROC curves—are computed during experiments.

Dependency List
Refer to requirements.txt, which typically includes:
numpy
pandas
scikit-learn
tensorflow or keras
pickle (for serialization)
jupyter 

Contributing
Contributions are welcome! Possible improvements:
Add a Streamlit or Flask interface for live inference.
Enhance the ANN model with hyperparameter tuning or dropout for regularization.
Add unit tests, CI pipelines, or Docker deployment support.

Workflow:
git checkout -b feature/YourFeature
# Make changes...
git commit -m "Add your changes"
git push origin feature/YourFeature

