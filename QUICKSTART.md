# 🚀 Quick Start Guide - AutoML Model Builder

Get your AutoML application running in 5 minutes!

## ⚡ Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python run.py
```

### 3. Open Your Browser
Go to: **http://localhost:5000**

## 🎯 Test with Sample Data

1. **Upload the sample dataset**: Use `sample_data/heart_disease_sample.csv`
2. **Select target column**: Choose `target` (heart disease prediction)
3. **Start training**: Click "Start Training"
4. **View results**: Compare model performances
5. **Make predictions**: Test with new patient data

## 📊 What You'll See

- **Dataset Analysis**: Automatic detection of features and task type
- **Model Training**: 7 algorithms trained simultaneously
- **Performance Comparison**: Interactive charts and metrics
- **Live Predictions**: Real-time inference on new data
- **Model Explanations**: Understand how models make decisions

## 🔧 Troubleshooting

### Common Issues:

**"Module not found" errors:**
```bash
pip install -r requirements.txt
```

**Port already in use:**
```bash
# Kill process on port 5000
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

**Memory issues with large datasets:**
- Reduce dataset size to <10,000 rows
- Close other applications
- Restart the application

## 📱 Features Overview

| Feature | Description |
|---------|-------------|
| 🎯 **AutoML** | Trains 7+ algorithms automatically |
| 📊 **Smart Analysis** | Detects task type and suggests target |
| 🚀 **Fast Training** | Parallel model training |
| 📈 **Visual Results** | Interactive charts and comparisons |
| 🔮 **Live Predictions** | Real-time inference |
| 💡 **Explainability** | SHAP and LIME explanations |
| 📥 **Model Export** | Download trained models |

## 🌟 Pro Tips

1. **Start Small**: Test with the sample dataset first
2. **Clean Data**: Ensure your CSV has headers and no missing values
3. **Target Selection**: Choose the column you want to predict
4. **Monitor Training**: Training takes 1-5 minutes for typical datasets
5. **Save Models**: Download your best models for production use

## 🆘 Need Help?

- Check the full [README.md](README.md)
- Review error messages in the terminal
- Ensure all dependencies are installed
- Try with the sample dataset first

---

**Ready to build your first ML model?** 🚀

Start the application and upload your data!
