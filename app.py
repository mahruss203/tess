from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model, scaler, label_encoder = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    prediction = model.predict(features)
    prediction_label = label_encoder.inverse_transform(prediction)
    return render_template('index.html', prediction_text=f'Prediksi: {prediction_label[0]}')

if __name__ == "__main__":
    app.run(debug=True)
