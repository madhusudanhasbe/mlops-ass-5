from flask import Flask, request, jsonify, render_template_string
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load model
model_path = 'models/model.pkl'
model_data = None

if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("Model file not found!")

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>MLOps Assignment 5 - Prediction App</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 50px auto;
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h1 { 
            color: #667eea;
            margin-bottom: 10px;
            text-align: center;
        }
        .status {
            text-align: center;
            padding: 15px;
            margin: 20px 0;
            border-radius: 8px;
            font-weight: bold;
        }
        .status.loaded { background: #d4edda; color: #155724; }
        .status.notloaded { background: #f8d7da; color: #721c24; }
        .form-group {
            margin: 20px 0;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 600;
        }
        input, textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            transition: border 0.3s;
        }
        input:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            width: 100%;
            padding: 15px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.3s;
        }
        button:hover {
            background: #5568d3;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .result h3 {
            color: #667eea;
            margin-bottom: 15px;
        }
        .prediction-output {
            background: white;
            padding: 15px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 MLOps Assignment 5</h1>
        <h2 style="text-align: center; color: #666; margin-bottom: 20px;">ML Prediction Application</h2>
        
        <div class="status {{ status_class }}">
            Model Status: {{ status }}
        </div>
        
        <form id="predictionForm">
            <div class="form-group">
                <label>📊 Enter Feature Values (comma-separated):</label>
                <textarea id="features" rows="3" placeholder="Example: 1.2, 3.4, 5.6, 7.8, 9.0"></textarea>
                <small style="color: #666;">Enter numerical values separated by commas</small>
            </div>
            <button type="submit">🔮 Predict</button>
        </form>
        
        <div id="result" class="result" style="display:none;">
            <h3>📈 Prediction Result:</h3>
            <div id="predictionOutput" class="prediction-output"></div>
        </div>

        <div class="footer">
            <p>MLOps Assignment 5 - Docker + AWS Deployment</p>
            <p>Roll No: 22070126061</p>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').onsubmit = async (e) => {
            e.preventDefault();
            const features = document.getElementById('features').value;
            
            if (!features.trim()) {
                alert('Please enter feature values!');
                return;
            }
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({features: features})
                });
                
                const data = await response.json();
                document.getElementById('result').style.display = 'block';
                document.getElementById('predictionOutput').textContent = JSON.stringify(data, null, 2);
            } catch (error) {
                alert('Error making prediction: ' + error.message);
            }
        };
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    status = "✅ Model Loaded Successfully" if model_data else "❌ Model Not Loaded"
    status_class = "loaded" if model_data else "notloaded"
    return render_template_string(HTML_TEMPLATE, status=status, status_class=status_class)

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_data is not None,
        'message': 'MLOps Assignment 5 - Application Running'
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not model_data:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        features = data.get('features', '')
        
        # Parse features
        feature_list = [float(x.strip()) for x in features.split(',')]
        feature_array = np.array([feature_list])
        
        # Scale features
        scaled_features = model_data['scaler'].transform(feature_array)
        
        # Make prediction
        prediction = model_data['model'].predict(scaled_features)[0]
        probability = model_data['model'].predict_proba(scaled_features)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': probability.tolist(),
            'features_received': feature_list,
            'num_features': len(feature_list),
            'message': 'Prediction successful ✅'
        })
    except ValueError as e:
        return jsonify({'error': f'Invalid input format: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Flask application...")
    print(f"Model loaded: {model_data is not None}")
    app.run(host='0.0.0.0', port=5000, debug=False)
