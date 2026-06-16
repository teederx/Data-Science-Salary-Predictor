Data Science Salary Predictor 🧪

A machine learning project that attempts to predict data science salaries using a neural network and Random Forest model, built with PyTorch and scikit-learn.


📁 Dataset

Source: Data Science Jobs Salaries via Kaggle

Features used after preprocessing:


experience_level — Entry, Mid, Senior, Executive (ordinal encoded)
employment_type — Full Time, Part Time, Contract (one-hot encoded)
company_size — Small, Medium, Large (ordinal encoded)
remote_ratio — Percentage of remote work


Target: salary_in_usd


🔧 Preprocessing Steps


Dropped irrelevant or high-cardinality columns: work_year, job_title, salary, salary_currency, employee_residence, company_location
Ordinal encoding (via .map()) for experience_level and company_size
One-hot encoding (via pd.get_dummies()) for employment_type
80/20 train-test split using random_split()
StandardScaler applied to features and labels — fitted on training data only to prevent data leakage



🏗️ Model Architecture — Neural Network

Built with PyTorch using fully connected layers with Batch Normalization:

Input (6 features)
    → Linear(6, 32) → BatchNorm → ReLU
    → Linear(32, 16) → BatchNorm → ReLU
    → Linear(16, 8) → BatchNorm → ReLU
    → Linear(8, 1) [no activation — regression output]

Training configuration:


Loss Function: HuberLoss (robust to salary outliers)
Optimizer: Adam (lr=0.0001)
Epochs: 10,000
Gradient clipping: max_norm=1.0
Weight initialization: Kaiming Normal



🌲 Model Architecture — Random Forest

pythonRandomForestRegressor(n_estimators=100, random_state=42)


📊 Results

ModelTrain R²Test R²Neural Network~0.33-1.42Random Forest0.350.26

⚠️ Why the Models Performed Poorly

Both models struggled to predict salaries accurately, and this is a direct result of dataset limitations, not the models themselves:


Too few meaningful features — after removing high-cardinality columns like job_title, employee_residence, and company_location, only 4 core features remained. These alone do not contain enough signal to accurately predict salary.
High salary variance — even within the same experience level and company size, salaries vary enormously. The features available cannot capture this variation.
Neural network overfitting — with 10,000 epochs and limited data, the neural network memorized training data but failed to generalize (Test R² = -1.42), demonstrating the classic effect of training too long on an inadequate dataset.
Random Forest generalized better — with a Test R² of 0.26 vs -1.42, Random Forest proved far more suitable for small tabular datasets, a well-known characteristic of tree-based models.


Key lesson: A model is only as good as its data. Features like specific job title, years of experience as a number, city/location, and company name would be needed to realistically push R² above 0.6.


🛠️ Requirements

torch
torchmetrics
scikit-learn
pandas
numpy
kagglehub

Install with:

bashpip install torch torchmetrics scikit-learn pandas numpy kagglehub


🚀 How to Run


Open the notebook in Google Colab
Run all cells in order from top to bottom
The dataset is downloaded automatically via kagglehub



📚 Concepts Demonstrated


Custom PyTorch Dataset and DataLoader
Batch Normalization and Kaiming weight initialization
Huber Loss for regression
Preventing data leakage with proper train/test scaling
Gradient clipping to prevent exploding gradients
R² Score as a regression evaluation metric
Comparison between deep learning and classical ML on tabular data
