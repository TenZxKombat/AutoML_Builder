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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
import cv2
from PIL import Image
import requests
from io import BytesIO
import base64
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern
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
        self.lemmatizer = WordNetLemmatizer()
        
        # Image processing components
        self.image_columns = []
        self.image_features = {}
        self.image_pca = None
        self.image_scaler = StandardScaler()
        
        # Download NLTK data if not present
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        
    def analyze_dataset(self, df):
        """Analyze the uploaded dataset to determine task type and data characteristics"""
        # Identify text columns (longer text content)
        text_columns = []
        categorical_columns = []
        
        for col in df.select_dtypes(include=['object']).columns:
            # Check if column contains image paths
            col_lower = col.lower()
            image_keywords = ['image', 'img', 'photo', 'picture', 'path', 'file']
            is_image_column = any(keyword in col_lower for keyword in image_keywords)
            
            if is_image_column:
                # This is an image column
                continue  # Skip image columns for now in text/categorical classification
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
        
        # Store text columns in the instance for later use
        self.text_columns = text_columns
        
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': categorical_columns,
            'text_columns': text_columns,
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
    
    def _preprocess_text(self, text_series):
        """Preprocess text data for feature extraction"""
        def clean_text(text):
            if pd.isna(text):
                return ""
            # Convert to string and lowercase
            text = str(text).lower()
            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        def lemmatize_text(text):
            if not text:
                return ""
            try:
                words = word_tokenize(text)
                lemmatized = [self.lemmatizer.lemmatize(word) for word in words]
                return ' '.join(lemmatized)
            except:
                return text
        
        # Clean and lemmatize text
        cleaned_text = text_series.apply(clean_text)
        lemmatized_text = cleaned_text.apply(lemmatize_text)
        return lemmatized_text
    
    def _extract_text_features(self, df, text_columns):
        """Extract features from text columns"""
        text_features = pd.DataFrame()
        
        for col in text_columns:
            if col in df.columns:
                # Preprocess text
                processed_text = self._preprocess_text(df[col])
                
                # Create TF-IDF vectorizer for this column
                tfidf = TfidfVectorizer(
                    max_features=100,  # Limit features to prevent overfitting
                    stop_words='english',
                    ngram_range=(1, 2),  # Unigrams and bigrams
                    min_df=2,  # Ignore terms that appear in less than 2 documents
                    max_df=0.95  # Ignore terms that appear in more than 95% of documents
                )
                
                # Fit and transform text
                tfidf_matrix = tfidf.fit_transform(processed_text)
                
                # Convert to DataFrame with meaningful column names
                feature_names = [f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
                col_features = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    columns=feature_names,
                    index=df.index
                )
                
                # Store vectorizer for later use
                self.tfidf_vectorizers[col] = tfidf
                
                # Add to text features
                text_features = pd.concat([text_features, col_features], axis=1)
        
        return text_features
    
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
            from skimage.feature import local_binary_pattern
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
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            # Return zero features if image processing fails
            return np.zeros(270)  # 256 (histogram) + 10 (LBP) + 1 (edge) + 3 (color) + 4 (stats)
    
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
                        features_list.append(np.zeros(270))
                
                # Convert to DataFrame
                feature_names = [f"{col}_img_{i}" for i in range(270)]
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
        """Preprocess the data for training with text processing capabilities"""
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
        
        # Extract text features
        text_features = pd.DataFrame()
        if len(text_cols) > 0:
            text_features = self._extract_text_features(df_cleaned, text_cols)
        
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
                               if col not in numeric_cols and col != target_column and col not in text_cols]
        if len(categorical_dummy_cols) > 0:
            final_features.append(df_encoded[categorical_dummy_cols])
        
        # Add text features
        if not text_features.empty:
            final_features.append(text_features)
        
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
        
        # Apply feature selection if we have many features
        if len(self.feature_columns) > 50:
            self._apply_feature_selection(df_final, target_column)
        
        return df_final
    
    def _apply_feature_selection(self, df, target_column):
        """Apply feature selection to reduce dimensionality"""
        X = df[self.feature_columns]
        y = df[target_column]
        
        # Use SelectKBest for feature selection
        if self.task_type == 'classification':
            selector = SelectKBest(score_func=f_classif, k=min(50, len(self.feature_columns)))
        else:
            selector = SelectKBest(score_func=f_regression, k=min(50, len(self.feature_columns)))
        
        X_selected = selector.fit_transform(X, y)
        
        # Update feature columns
        selected_features = [self.feature_columns[i] for i in selector.get_support(indices=True)]
        self.feature_columns = selected_features
        
        # Store selector for later use
        self.text_feature_selector = selector
    
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
    
    def get_model_explanations(self, X_sample, model_name):
        """Get model explanations using SHAP and LIME"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]['model']
        explanations = {}
        
        try:
            # SHAP explanations
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.LinearExplainer(model, X_sample)
                shap_values = explainer.shap_values(X_sample)
                explanations['shap_values'] = shap_values.tolist() if isinstance(shap_values, np.ndarray) else [sv.tolist() for sv in shap_values]
            
            # LIME explanations
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_sample.values, 
                feature_names=X_sample.columns,
                class_names=['Class 0', 'Class 1'] if self.task_type == 'classification' else None,
                mode='classification' if self.task_type == 'classification' else 'regression'
            )
            
            lime_exp = explainer.explain_instance(
                X_sample.iloc[0].values, 
                model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                num_features=min(10, len(X_sample.columns))
            )
            
            explanations['lime'] = {
                'feature_weights': lime_exp.as_list(),
                'prediction': lime_exp.predicted_value
            }
            
        except Exception as e:
            explanations['error'] = str(e)
        
        return explanations

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
        
        if file and (file.filename.endswith('.csv') or file.filename.lower().endswith(('.jpg', '.jpeg', '.png'))):
            if file.filename.endswith('.csv'):
                # Read the CSV file
                df = pd.read_csv(file)
                # Limit to 15 columns maximum
                if df.shape[1] > 15:
                    df = df.iloc[:, :15]
                current_dataset = df
                
                # Analyze the dataset
                analysis = automl.analyze_dataset(df)
                # Compute top 5 important columns for frontend display/target selection
                def _estimate_importance(local_df):
                    scores = {}
                    for col in local_df.columns:
                        series = local_df[col]
                        if pd.api.types.is_numeric_dtype(series):
                            # Higher variance implies higher potential importance
                            scores[col] = float(np.nan_to_num(series.var(ddof=0)))
                        elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                            # Use average string length as a proxy; if categorical, use inverse of unique count
                            try:
                                avg_len = series.dropna().astype(str).str.len().mean()
                            except Exception:
                                avg_len = 0.0
                            nunique = series.nunique(dropna=True)
                            # Combine both signals
                            scores[col] = float((avg_len or 0.0) + (1.0 / (nunique + 1)))
                        else:
                            # Fallback: number of non-nulls
                            scores[col] = float(series.notna().sum())
                    return scores
                importance_scores = _estimate_importance(df)
                sorted_cols = sorted(importance_scores.keys(), key=lambda c: importance_scores[c], reverse=True)
                top_features = sorted_cols[:5]
                analysis['top_features'] = top_features
                analysis['limited_columns'] = list(df.columns)
                # Restrict columns exposed to the frontend to only the top 5
                analysis['columns'] = top_features
                current_task_type = analysis['task_type']
                
                return jsonify({
                    'success': True,
                    'analysis': analysis,
                    'message': f'Dataset uploaded successfully! Shape: {df.shape}'
                })
            else:
                # Handle individual image upload
                # For now, create a simple dataset with the image
                image_filename = secure_filename(file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
                file.save(image_path)
                
                # Create a simple dataset with the image
                df = pd.DataFrame({
                    'image_path': [image_path],
                    'label': ['unknown']  # Default label
                })
                current_dataset = df
                
                # Analyze the dataset
                analysis = automl.analyze_dataset(df)
                # For image-only case, limit to available columns (<= 2) and set top_features accordingly
                analysis['top_features'] = list(df.columns)[:5]
                analysis['limited_columns'] = list(df.columns)
                analysis['columns'] = analysis['top_features']
                current_task_type = analysis['task_type']
                
                return jsonify({
                    'success': True,
                    'analysis': analysis,
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

@app.route('/predict', methods=['POST'])
def make_prediction():
    try:
        data = request.get_json()
        input_data = data.get('input_data')
        model_name = data.get('model_name', 'best')
        
        if current_dataset is None:
            return jsonify({'error': 'No dataset available'}), 400
        
        # Get the model
        if model_name == 'best' and automl.best_model:
            model = automl.best_model
        elif model_name in current_models:
            model = current_models[model_name]['model']
        else:
            return jsonify({'error': 'Model not found'}), 400
        
        # Preprocess input data
        input_df = pd.DataFrame([input_data])
        
        # Apply the same preprocessing as training
        if len(automl.feature_columns) > 0:
            # Handle text features
            if hasattr(automl, 'text_columns') and automl.text_columns:
                text_features = automl._extract_text_features(input_df, automl.text_columns)
                # Remove original text columns and add processed features
                input_df = input_df.drop(columns=automl.text_columns, errors='ignore')
                input_df = pd.concat([input_df, text_features], axis=1)
            
            # Handle categorical features
            categorical_cols = input_df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
            
            # Ensure all feature columns exist
            for col in automl.feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Reorder columns to match training data
            input_df = input_df[automl.feature_columns]
            
            # Apply feature selection if used during training
            if hasattr(automl, 'text_feature_selector') and automl.text_feature_selector:
                input_df = pd.DataFrame(
                    automl.text_feature_selector.transform(input_df),
                    columns=automl.feature_columns
                )
            
            # Scale all features
            input_df = automl.scaler.transform(input_df)
        
        # Make prediction
        if automl.task_type == 'classification':
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None
            
            # Decode prediction if label encoder was used
            if hasattr(automl, 'label_encoder') and automl.label_encoder.classes_ is not None:
                prediction = automl.label_encoder.inverse_transform([prediction])[0]
            
            return jsonify({
                'success': True,
                'prediction': str(prediction),
                'probability': probability.tolist() if probability is not None else None,
                'task_type': 'classification'
            })
        else:
            prediction = model.predict(input_df)[0]
            return jsonify({
                'success': True,
                'prediction': float(prediction),
                'task_type': 'regression'
            })
            
    except Exception as e:
        return jsonify({'error': f'Error making prediction: {str(e)}'}), 500

@app.route('/explanations/<model_name>')
def get_explanations(model_name):
    try:
        if current_dataset is None or model_name not in current_models:
            return jsonify({'error': 'Model not found'}), 400
        
        # Get a sample from the dataset for explanations
        df_processed = automl.preprocess_data(current_dataset, automl.target_column)
        X_sample = df_processed[automl.feature_columns].head(1)
        
        explanations = automl.get_model_explanations(X_sample, model_name)
        return jsonify({'success': True, 'explanations': explanations})
        
    except Exception as e:
        return jsonify({'error': f'Error getting explanations: {str(e)}'}), 500

@app.route('/download_model/<model_name>')
def download_model(model_name):
    try:
        if model_name not in current_models:
            return jsonify({'error': 'Model not found'}), 400
        
        model = current_models[model_name]['model']
        model_path = os.path.join(app.config['MODELS_FOLDER'], f'{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib')
        
        # Save the model
        joblib.dump(model, model_path)
        
        return send_file(model_path, as_attachment=True, download_name=f'{model_name}.joblib')
        
    except Exception as e:
        return jsonify({'error': f'Error downloading model: {str(e)}'}), 500

@app.route('/get_metrics')
def get_metrics():
    return jsonify(current_metrics)

@app.route('/get_dataset_info')
def get_dataset_info():
    if current_dataset is not None:
        return jsonify({
            'shape': current_dataset.shape,
            'columns': list(current_dataset.columns),
            'task_type': current_task_type
        })
    return jsonify({'error': 'No dataset available'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
