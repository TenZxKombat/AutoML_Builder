from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Global variables
current_dataset = None

@app.route('/')
def home():
    return """
    <h1>AutoML Test App</h1>
    <p>This is a minimal test version to debug JSON issues.</p>
    <p>Test endpoints:</p>
    <ul>
        <li><a href="/test">GET /test</a></li>
        <li>POST /upload (with CSV file)</li>
        <li>POST /train (with JSON)</li>
        <li>POST /predict (with JSON)</li>
    </ul>
    """

@app.route('/test')
def test():
    return jsonify({'status': 'ok', 'message': 'Test endpoint working'})

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            global current_dataset
            current_dataset = df
            
            return jsonify({
                'success': True,
                'message': f'File uploaded! Shape: {df.shape}',
                'columns': list(df.columns)
            })
        else:
            return jsonify({'error': 'Please upload a CSV file'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Upload error: {str(e)}'}), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        # Check if we have JSON data
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No JSON data received'}), 400
        
        target_column = data.get('target_column')
        if not target_column:
            return jsonify({'error': 'Missing target_column in JSON'}), 400
        
        if current_dataset is None:
            return jsonify({'error': 'No dataset uploaded yet'}), 400
        
        if target_column not in current_dataset.columns:
            return jsonify({'error': f'Column "{target_column}" not found in dataset'}), 400
        
        return jsonify({
            'success': True,
            'message': f'Training setup successful for target: {target_column}',
            'dataset_shape': current_dataset.shape,
            'target_column': target_column
        })
        
    except Exception as e:
        return jsonify({'error': f'Training error: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if we have JSON data
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No JSON data received'}), 400
        
        input_data = data.get('input_data')
        if not input_data:
            return jsonify({'error': 'Missing input_data in JSON'}), 400
        
        if current_dataset is None:
            return jsonify({'error': 'No dataset uploaded yet'}), 400
        
        return jsonify({
            'success': True,
            'message': 'Prediction endpoint working',
            'received_data': input_data,
            'dataset_columns': list(current_dataset.columns)
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/debug')
def debug():
    """Debug endpoint to check app state"""
    return jsonify({
        'app_running': True,
        'dataset_loaded': current_dataset is not None,
        'dataset_shape': current_dataset.shape if current_dataset is not None else None,
        'dataset_columns': list(current_dataset.columns) if current_dataset is not None else []
    })

if __name__ == '__main__':
    print("🚀 Starting Simple AutoML Test App...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🧪 Test the endpoints to see JSON handling...")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
