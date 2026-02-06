## Building the Fetal Health Prediction Model for Time Series Analytics Microservice

The fetal health prediction model is built using machine learning techniques to classify fetal health conditions based on Cardiotocography (CTG) data. The model provides real-time predictions for Normal, Suspect, and Pathological conditions with confidence scores.

### Dataset and Model Details

- **Dataset**: Fetal Health Classification dataset from UCI ML Repository
- **Features**: 21 CTG parameters including baseline fetal heart rate, accelerations, decelerations, and variability measures
- **Algorithm**: XGBoost Classifier with feature engineering
- **Performance**: 93.43% accuracy on test data
- **Output**: Three-class classification with risk assessment and confidence scores

### Model Training Setup

To retrain or modify the fetal health prediction model:

```bash
# 1. Navigate to the training directory
cd edge-ai-suites/manufacturing-ai-suite/industrial-edge-insights-time-series/apps/fetal-health-prediction/training

# 2. Create and activate virtual environment
python3 -m venv ~/fetal_health_env
source ~/fetal_health_env/bin/activate

# 3. Install required packages
pip3 install -r requirements.txt

# 4. Launch Jupyter notebook (if available)
jupyter-notebook --ip=0.0.0.0 --port=8888 --no-browser

# 5. Open the training notebook
# http://<system_ip>:8888/notebooks/fetal_health_training.ipynb
```

### Model Architecture

The fetal health prediction system includes:

1. **Feature Engineering Pipeline**:
   - Risk score calculation from decelerations
   - Variability risk assessment
   - Heart rate baseline risk indicators
   - Histogram-based statistical features
   - Combined risk factors (27 features total)

2. **XGBoost Classifier**:
   - Optimized hyperparameters (max_depth=6, learning_rate=0.1)
   - CPU-based inference for edge deployment
   - Real-time processing capability

3. **Risk Assessment Framework**:
   - **Normal (0)**: Low risk - routine monitoring
   - **Suspect (1)**: Medium risk - enhanced monitoring
   - **Pathological (2)**: High risk - immediate medical attention

### Integration with Time Series Analytics

The trained model (`fetal_health_predictor.pkl`) is deployed as a UDF (User Defined Function) that:

- Processes incoming CTG data streams in real-time
- Applies feature engineering and prediction pipeline
- Outputs risk classifications with confidence scores
- Triggers alerts for high-risk conditions (confidence > 80%)
- Provides detailed logging for clinical decision support

### Model Files

- **fetal_health_predictor.pkl**: Complete trained model pipeline
- **fetal_health_predictor.py**: UDF implementation for edge deployment
- **fetal_health_predictor.tick**: TICK script for Kapacitor integration
- **requirements.txt**: Python dependencies for model training and inference

### Clinical Usage

The model is designed for clinical decision support and should always be used in conjunction with medical professional judgment. It processes standard CTG measurements and provides:

- Real-time fetal health status classification
- Confidence scores for prediction reliability  
- Alert notifications for high-risk conditions
- Comprehensive logging for medical record integration