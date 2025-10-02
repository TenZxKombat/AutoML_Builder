from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'test-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODELS_FOLDER'] = 'models'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)

# Global variables
current_dataset = None
current_task_type = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_dataset, current_task_type
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.endswith('.csv'):
            # Read the CSV file
            df = pd.read_csv(file)
            current_dataset = df
            
            # Simple analysis
            analysis = {
                'shape': df.shape,
                'columns': list(df.columns),
                'task_type': 'classification' if df.select_dtypes(include=['object']).shape[1] > 0 else 'regression'
            }
            current_task_type = analysis['task_type']
            
            return jsonify({
                'success': True,
                'analysis': analysis,
                'message': f'Dataset uploaded successfully! Shape: {df.shape}'
            })
        else:
            return jsonify({'error': 'Please upload a CSV file'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/train', methods=['POST'])
def train_models():
    global current_dataset
    
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No JSON data received'}), 400
            
        target_column = data.get('target_column')
        if not target_column:
            return jsonify({'error': 'No target_column provided'}), 400
        
        if current_dataset is None:
            return jsonify({'error': 'No dataset uploaded'}), 400
        
        # Simple response for testing
        return jsonify({
            'success': True,
            'message': f'Training started with target column: {target_column}',
            'dataset_shape': current_dataset.shape
        })
        
    except Exception as e:
        return jsonify({'error': f'Error in training: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def make_prediction():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No JSON data received'}), 400
            
        input_data = data.get('input_data')
        if not input_data:
            return jsonify({'error': 'No input_data provided'}), 400
        
        if current_dataset is None:
            return jsonify({'error': 'No dataset available'}), 400
        
        # Simple response for testing
        return jsonify({
            'success': True,
            'message': 'Prediction endpoint working',
            'received_data': input_data
        })
        
    except Exception as e:
        return jsonify({'error': f'Error in prediction: {str(e)}'}), 500

@app.route('/test')
def test():
    return jsonify({'status': 'ok', 'message': 'Test endpoint working'})

if __name__ == '__main__':
    print("🚀 Starting Test AutoML App...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🧪 Test endpoints:")
    print("   - GET  /test")
    print("   - POST /upload (with CSV file)")
    print("   - POST /train (with JSON: {\"target_column\": \"column_name\"})")
    print("   - POST /predict (with JSON: {\"input_data\": {\"feature\": \"value\"}})")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
