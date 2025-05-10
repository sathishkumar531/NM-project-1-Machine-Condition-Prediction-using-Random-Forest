
# Sathishkumar D

**2nd Year, Mechanical Engineering**
**ARM College of Engineering & Technology**
**Course: Data Analysis in Mechanical Engineering**

---

## Machine Condition Prediction Using Random Forest

This project is focused on predicting the health condition of a machine using a machine learning model called **Random Forest Classifier**. The prediction is based on various sensor readings such as temperature, vibration, oil quality, RPM, and more. The goal is to make it easier to understand whether a machine is operating normally or if there's a possible fault.

---

### Setup Instructions

Before running the code, make sure all required Python packages are installed. You can do that by running the following command:

```bash
pip install -r requirements.txt
```

---

### Files Included

To make predictions, a few important files need to be available in your working directory:

* **`random_forest_model.pkl`** – This is the trained Random Forest model used for prediction.
* **`scaler.pkl`** – A scaler that was used during training to normalize the input values.
* **`selected_features.pkl`** – A file that lists the exact features used to train the model, so the input stays consistent.

---

### How the Prediction Process Works

Here’s a step-by-step breakdown of how the prediction system works:

1. **Loading Resources**

   * The trained model, scaler, and feature list are loaded using `joblib`.

2. **Preparing New Input Data**

   * A new data point is created using `pandas.DataFrame`, with all the required features in a single row.

3. **Data Preprocessing**

   * The scaler is used to normalize the new input so that it matches the data the model was trained on.

4. **Making a Prediction**

   * The `.predict()` method tells you the machine's condition (normal or faulty).
   * The `.predict_proba()` method gives the probability score for each class.

---

### Example Python Script for Prediction

You can use the following script to run predictions on new input data:

```python
import joblib
import pandas as pd

# Load model and preprocessing tools
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')

# Replace this with actual input values
new_data = pd.DataFrame([{
    'Temperature_C': 75,
    'Vibration_mm_s': 2.5,
    'Oil_Quality_Index': 88,
    'RPM': 1500,
    'Pressure_bar': 5.2,
    'Shaft_Misalignment_deg': 0.3,
    'Noise_dB': 70,
    'Load_%': 85,
    'Power_kW': 12.5
}])

# Ensure correct feature order
new_data = new_data[selected_features]

# Scale the data
scaled_data = scaler.transform(new_data)

# Get prediction
prediction = model.predict(scaled_data)
prediction_proba = model.predict_proba(scaled_data)

print("Predicted Class:", prediction[0])
print("Prediction Probabilities:", prediction_proba[0])
```

---

### Important Tips

* Your input data must include **all the exact same features** as the model was trained with.
* Make sure the feature values are within the typical range of the training data.
* The **order of the columns** is very important — do not change it.

---

### Optional: Training the Model Again

If you ever need to retrain the model with new data:

* Use the same preprocessing steps (especially scaling).
* Keep the feature selection consistent.
* Save the new model and files using `joblib` again.

---

### Possible Applications

This type of prediction model can be used in:

* Identifying faulty machines in factories.
* Preventive maintenance based on sensor data.
* Integrating into IoT systems for live condition monitoring.
