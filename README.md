# AutoML Model Builder

An intelligent, automated machine learning model builder that trains and compares multiple ML models automatically with zero coding required.

## 🚀 Features

### Core Functionality
- **Automatic Dataset Analysis**: Detects features, labels, data types, and null values
- **Smart Task Detection**: Automatically determines if it's classification or regression
- **AutoML Training**: Trains multiple algorithms simultaneously
- **Performance Comparison**: Compares models using cross-validation
- **Best Model Selection**: Automatically selects and saves the best performing model
- **Live Predictions**: Make real-time predictions through the web interface
- **Model Explainability**: SHAP and LIME explanations for model decisions

### Supported Algorithms

#### Classification
- Logistic Regression
- Random Forest
- Decision Tree
- Support Vector Machine (SVM)
- XGBoost
- LightGBM
- CatBoost

#### Regression
- Linear Regression
- Random Forest
- Decision Tree
- Support Vector Regression (SVR)
- XGBoost
- LightGBM
- CatBoost

### Data Preprocessing
- **Missing Value Handling**: Automatic imputation for numeric and categorical data
- **Feature Encoding**: One-hot encoding for categorical variables
- **Feature Scaling**: Standardization of numeric features
- **Label Encoding**: Automatic encoding of target variables

## 🏥 Real-World Use Cases

### 1. Healthcare: Disease Prediction
- **Use Case**: Predict heart disease risk from patient data
- **Input**: Age, blood pressure, cholesterol, etc.
- **Output**: Disease probability
- **Real Example**: Hospital triage systems without ML experts

### 2. Human Resources: Employee Attrition
- **Use Case**: Predict employee retention risk
- **Input**: Job satisfaction, salary, performance metrics
- **Output**: Attrition probability
- **Real Example**: IBM Watson and Google AutoML powered tools

### 3. Finance: Credit Risk Assessment
- **Use Case**: Predict loan default probability
- **Input**: Income, credit history, employment status
- **Output**: Default risk score

### 4. Marketing: Customer Churn Prediction
- **Use Case**: Predict customer churn
- **Input**: Purchase history, engagement metrics, demographics
- **Output**: Churn probability

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd automl-builder
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
python app.py
```

4. **Open your browser** and go to `http://localhost:5000`

## 📊 How to Use

### Step 1: Upload Dataset
- Drag and drop a CSV file or click to browse
- The system automatically analyzes your data
- Detects task type (classification/regression)
- Suggests the best target column

### Step 2: Train Models
- Select your target column (what you want to predict)
- Click "Start Training"
- The system trains 7 different algorithms simultaneously
- Shows real-time training progress

### Step 3: Compare Results
- View performance metrics for all models
- Interactive charts showing model comparison
- Download any trained model
- Identify the best performing algorithm

### Step 4: Make Predictions
- Input new data values
- Get instant predictions
- View prediction probabilities (for classification)
- Use any trained model or the best model

### Step 5: Understand Models
- Get SHAP values for feature importance
- LIME explanations for individual predictions
- Understand how each feature contributes to predictions

## 🔧 Technical Details

### Architecture
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **ML Libraries**: Scikit-learn, XGBoost, LightGBM, CatBoost
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Explainability**: SHAP, LIME

### Data Flow
1. **Upload** → CSV file processing
2. **Analysis** → Automatic feature detection and task classification
3. **Preprocessing** → Data cleaning, encoding, scaling
4. **Training** → Multiple model training with cross-validation
5. **Evaluation** → Performance metrics calculation
6. **Selection** → Best model identification
7. **Deployment** → Model download and live predictions

### Performance Metrics

#### Classification
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

#### Regression
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

## 📁 Project Structure

```
automl-builder/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/            # HTML templates
│   └── index.html       # Main interface
├── uploads/             # Temporary file storage
├── models/              # Saved trained models
└── README.md            # This file
```

## 🌟 Key Benefits

1. **No Coding Required**: Fully automated ML pipeline
2. **Time Saving**: Train multiple models in parallel
3. **Best Model Selection**: Automatic selection based on performance
4. **Production Ready**: Download trained models for deployment
5. **Explainable AI**: Understand model decisions
6. **User Friendly**: Beautiful, intuitive web interface
7. **Scalable**: Handle datasets up to 16MB
8. **Cross-Platform**: Works on Windows, Mac, and Linux

## 🔒 Security Features

- File size limits (16MB max)
- Secure file handling
- Input validation
- Error handling and logging

## 🚀 Future Enhancements

- **Cloud Deployment**: Deploy models to cloud platforms
- **API Integration**: RESTful API for model serving
- **Advanced Algorithms**: Deep learning models
- **Hyperparameter Tuning**: Automated hyperparameter optimization
- **Feature Engineering**: Automatic feature creation
- **Model Monitoring**: Performance tracking over time
- **Multi-language Support**: Interface in multiple languages

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information
4. Include your dataset structure and error messages

## 🎯 Quick Start Example

1. **Prepare your data**: Save as CSV with headers
2. **Upload**: Drag and drop your CSV file
3. **Train**: Select target column and start training
4. **Compare**: View results and download best model
5. **Predict**: Input new data for live predictions

## 📈 Performance Tips

- **Dataset Size**: Optimal for datasets with 100-10,000 rows
- **Feature Count**: Works best with 5-50 features
- **Memory**: Ensure sufficient RAM for large datasets
- **Training Time**: Expect 1-5 minutes for typical datasets

---

**Built with ❤️ for the ML community**

Transform your data into insights with zero coding required!
