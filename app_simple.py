from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
import traceback
from werkzeug.utils import secure_filename
import plotly.graph_objs as go
import plotly.utils
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, mean_squared_error, r2_score,
    roc_auc_score, roc_curve
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import cv2
from PIL import Image
import requests
from io import BytesIO
import base64
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODELS_FOLDER'] = 'models'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)

# Global variables to store current dataset and models
current_dataset = None
current_models = {}
current_task_type = None
current_metrics = {}

class AutoMLBuilder:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.task_type = None
        self.feature_columns = []
        self.target_column = None
        self.preprocessor = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.label_encoder = LabelEncoder()
        
        # Text processing components
        self.text_columns = []
        self.tfidf_vectorizers = {}
        self.text_feature_selector = None
        
        # Image processing components
        self.image_columns = []
        self.image_features = {}
        self.image_pca = None
        self.image_scaler = StandardScaler()
        
    def analyze_dataset(self, df):
        """Analyze the uploaded dataset to determine task type and data characteristics"""
        # Identify text columns (longer text content)
        text_columns = []
        categorical_columns = []
        image_columns = []
        
        for col in df.select_dtypes(include=['object']).columns:
            col_lower = col.lower()
            image_keywords = ['image', 'img', 'photo', 'picture', 'path', 'file']
            is_image_column = any(keyword in col_lower for keyword in image_keywords)
            
            if is_image_column:
                image_columns.append(col)
            else:
                # Check if column contains text (average length > 10 characters or contains common text keywords)
                avg_length = df[col].dropna().astype(str).str.len().mean()
                
                # Check for common text column names
                text_keywords = ['text', 'description', 'review', 'comment', 'feedback', 'content', 'message', 'note', 'summary']
                is_text_column = any(keyword in col_lower for keyword in text_keywords)
                
                if avg_length > 10 or is_text_column:
                    text_columns.append(col)
                else:
                    categorical_columns.append(col)
        
        # Store columns in the instance for later use
        self.text_columns = text_columns
        self.image_columns = image_columns
        
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': categorical_columns,
            'text_columns': text_columns,
            'image_columns': image_columns,
            'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
        }
        
        # Determine task type
        if len(analysis['numeric_columns']) == 0:
            self.task_type = 'classification'
        else:
            # Check if target column is categorical or numeric
            potential_targets = analysis['numeric_columns'] + analysis['categorical_columns']
            for col in potential_targets:
                if df[col].nunique() <= 20:  # Likely classification
                    self.task_type = 'classification'
                    break
                else:
                    self.task_type = 'regression'
        
        analysis['task_type'] = self.task_type
        analysis['suggested_target'] = self._suggest_target_column(df, analysis)
        
        return analysis
    
    def _preprocess_image(self, image_path):
        """Preprocess a single image for feature extraction"""
        try:
            # Load image
            if image_path.startswith('http'):
                response = requests.get(image_path)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to standard size (224x224 for most models)
            img = img.resize((224, 224))
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Convert to grayscale for simpler feature extraction
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Extract basic features
            features = []
            
            # Histogram features
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            features.extend(hist.flatten())
            
            # Texture features using Local Binary Pattern
            lbp = local_binary_pattern(gray, 8, 1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
            features.extend(lbp_hist)
            
            # Edge features
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            
            # Color features (mean RGB values)
            mean_r = np.mean(img_array[:, :, 0])
            mean_g = np.mean(img_array[:, :, 1])
            mean_b = np.mean(img_array[:, :, 2])
            features.extend([mean_r, mean_g, mean_b])
            
            # Statistical features
            features.extend([
                np.mean(gray),
                np.std(gray),
                np.var(gray),
                np.median(gray)
            ])
            
            features_array = np.array(features)
            print(f"Extracted {len(features_array)} features from image {image_path}")
            return features_array
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            # Return zero features if image processing fails
            return np.zeros(274)  # Updated to match actual feature count
    
    def _extract_image_features(self, df, image_columns):
        """Extract features from image columns"""
        image_features = pd.DataFrame()
        
        for col in image_columns:
            if col in df.columns:
                print(f"Processing images from column: {col}")
                features_list = []
                
                for idx, image_path in enumerate(df[col]):
                    if pd.notna(image_path):
                        features = self._preprocess_image(image_path)
                        features_list.append(features)
                    else:
                        # Handle missing images
                        features_list.append(np.zeros(274))
                
                # Convert to DataFrame
                feature_names = [f"{col}_img_{i}" for i in range(274)]
                col_features = pd.DataFrame(
                    features_list,
                    columns=feature_names,
                    index=df.index
                )
                
                # Store features
                self.image_features[col] = col_features
                image_features = pd.concat([image_features, col_features], axis=1)
        
        return image_features
    
    def _suggest_target_column(self, df, analysis):
        """Suggest the best target column based on data characteristics"""
        if self.task_type == 'classification':
            # For classification, prefer columns with fewer unique values
            categorical_cols = analysis['categorical_columns']
            if categorical_cols:
                return min(categorical_cols, key=lambda x: df[x].nunique())
            else:
                # If no categorical columns, use the column with fewest unique values
                return min(analysis['numeric_columns'], key=lambda x: df[x].nunique())
        else:
            # For regression, prefer numeric columns
            return analysis['numeric_columns'][-1] if analysis['numeric_columns'] else None
    
    def preprocess_data(self, df, target_column):
        """Preprocess the data for training with image processing capabilities"""
        self.target_column = target_column
        self.feature_columns = [col for col in df.columns if col != target_column]
        
        # Identify column types
        numeric_cols = df[self.feature_columns].select_dtypes(include=[np.number]).columns
        categorical_cols = []
        text_cols = []
        image_cols = []
        
        # Re-identify text vs categorical vs image columns
        for col in df[self.feature_columns].select_dtypes(include=['object']).columns:
            col_lower = col.lower()
            image_keywords = ['image', 'img', 'photo', 'picture', 'path', 'file']
            is_image_column = any(keyword in col_lower for keyword in image_keywords)
            
            if is_image_column:
                image_cols.append(col)
            else:
                avg_length = df[col].dropna().astype(str).str.len().mean()
                if avg_length > 10:
                    text_cols.append(col)
                else:
                    categorical_cols.append(col)
        
        self.text_columns = text_cols
        self.image_columns = image_cols
        
        # Handle missing values
        df_cleaned = df.copy()
        
        # Fill numeric columns with mean
        if len(numeric_cols) > 0:
            df_cleaned[numeric_cols] = self.imputer.fit_transform(df_cleaned[numeric_cols])
        
        # Fill categorical columns with mode
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
        
        # Fill text columns with empty string
        if len(text_cols) > 0:
            for col in text_cols:
                df_cleaned[col] = df_cleaned[col].fillna('')
        
        # Extract image features
        image_features = pd.DataFrame()
        if len(image_cols) > 0:
            image_features = self._extract_image_features(df_cleaned, image_cols)
        
        # Encode categorical features
        if len(categorical_cols) > 0:
            df_encoded = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)
        else:
            df_encoded = df_cleaned.copy()
        
        # Combine all features
        final_features = []
        
        # Add numeric features
        if len(numeric_cols) > 0:
            final_features.append(df_encoded[numeric_cols])
        
        # Add categorical features (after one-hot encoding)
        categorical_dummy_cols = [col for col in df_encoded.columns 
                               if col not in numeric_cols and col != target_column and col not in text_cols and col not in image_cols]
        if len(categorical_dummy_cols) > 0:
            final_features.append(df_encoded[categorical_dummy_cols])
        
        # Add image features
        if not image_features.empty:
            final_features.append(image_features)
        
        # Combine all features
        if final_features:
            X_combined = pd.concat(final_features, axis=1)
        else:
            X_combined = pd.DataFrame(index=df.index)
        
        # Update feature columns
        self.feature_columns = list(X_combined.columns)
        
        # Scale all features
        if len(self.feature_columns) > 0:
            X_scaled = self.scaler.fit_transform(X_combined)
            X_combined = pd.DataFrame(X_scaled, columns=self.feature_columns, index=df.index)
        
        # Add target column
        df_final = df_cleaned[[target_column]].copy()
        df_final = pd.concat([df_final, X_combined], axis=1)
        
        # Encode target variable for classification
        if self.task_type == 'classification':
            df_final[target_column] = self.label_encoder.fit_transform(df_final[target_column])
        
        return df_final
    
    def get_models(self):
        """Get the appropriate models based on task type"""
        if self.task_type == 'classification':
            return {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'SVM': SVC(random_state=42, probability=True),
                'XGBoost': xgb.XGBClassifier(random_state=42),
                'LightGBM': lgb.LGBMClassifier(random_state=42),
                'CatBoost': cb.CatBoostClassifier(random_state=42, verbose=False)
            }
        else:
            return {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'SVR': SVR(),
                'XGBoost': xgb.XGBRegressor(random_state=42),
                'LightGBM': lgb.LGBMRegressor(random_state=42),
                'CatBoost': cb.CatBoostRegressor(random_state=42, verbose=False)
            }
    
    def train_models(self, X, y):
        """Train multiple models and compare their performance"""
        models = self.get_models()
        results = {}
        
        # Handle small datasets
        if len(X) < 5:
            # For very small datasets, use all data for training and testing
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if self.task_type == 'classification' else None
            )
        
        for name, model in models.items():
            try:
                print(f"Training {name}...")
                
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                if self.task_type == 'classification':
                    metrics = self._calculate_classification_metrics(y_test, y_pred, y_pred_proba)
                else:
                    metrics = self._calculate_regression_metrics(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                # Update best model
                if self.task_type == 'classification':
                    score = metrics['f1_score']
                else:
                    score = metrics['r2_score']
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = model
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        self.models = results
        return results
    
    def _calculate_classification_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except:
                metrics['roc_auc'] = None
        
        return metrics
    
    def _calculate_regression_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2_score': r2_score(y_true, y_pred)
        }

# Global AutoML instance
automl = AutoMLBuilder()

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
        
        print(f"Uploaded file: {file.filename}")
        print(f"File type: {file.content_type}")
        
        if file and (file.filename.endswith('.csv') or any(file.filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])):
            if file.filename.endswith('.csv'):
                # Read the CSV file
                df = pd.read_csv(file)
                current_dataset = df
                
                # Analyze the dataset
                analysis = automl.analyze_dataset(df)
                current_task_type = analysis['task_type']
                
                return jsonify({
                    'success': True,
                    'analysis': analysis,
                    'message': f'Dataset uploaded successfully! Shape: {df.shape}'
                })
            else:
                # Handle individual image upload
                print("Processing individual image upload...")
                image_filename = secure_filename(file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
                print(f"Saving image to: {image_path}")
                file.save(image_path)
                print(f"Image saved successfully")
                
                # Create a simple dataset with the image
                df = pd.DataFrame({
                    'image_path': [image_path],
                    'label': ['unknown']  # Default label
                })
                print(f"Created dataset: {df}")
                current_dataset = df
                
                # Analyze the dataset
                print("Analyzing dataset...")
                analysis = automl.analyze_dataset(df)
                current_task_type = analysis['task_type']
                print(f"Analysis result: {analysis}")
                
                # Clean up analysis for JSON serialization
                clean_analysis = {
                    'shape': analysis['shape'],
                    'columns': analysis['columns'],
                    'dtypes': {str(k): str(v) for k, v in analysis['dtypes'].items()},
                    'null_counts': analysis['null_counts'],
                    'numeric_columns': analysis['numeric_columns'],
                    'categorical_columns': analysis['categorical_columns'],
                    'text_columns': analysis['text_columns'],
                    'image_columns': analysis['image_columns'],
                    'memory_usage': analysis['memory_usage'],
                    'task_type': analysis['task_type'],
                    'suggested_target': analysis['suggested_target']
                }
                
                return jsonify({
                    'success': True,
                    'analysis': clean_analysis,
                    'message': f'Image uploaded successfully! You can now add more images or train with this one.'
                })
        else:
            return jsonify({'error': 'Please upload a CSV file or image (JPG, PNG)'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/train', methods=['POST'])
def train_models():
    global current_dataset, current_models, current_metrics
    
    try:
        data = request.get_json()
        target_column = data.get('target_column')
        
        if current_dataset is None:
            return jsonify({'error': 'No dataset uploaded'}), 400
        
        # Preprocess the data
        df_processed = automl.preprocess_data(current_dataset, target_column)
        
        # Prepare features and target
        X = df_processed[automl.feature_columns]
        y = df_processed[target_column]
        
        # Train models
        results = automl.train_models(X, y)
        current_models = results
        
        # Extract metrics for display
        current_metrics = {}
        for name, result in results.items():
            if 'error' not in result:
                current_metrics[name] = result['metrics']
        
        return jsonify({
            'success': True,
            'results': current_metrics,
            'best_model': automl.best_model.__class__.__name__ if automl.best_model else None,
            'message': 'Models trained successfully!'
        })
        
    except Exception as e:
        return jsonify({'error': f'Error training models: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
