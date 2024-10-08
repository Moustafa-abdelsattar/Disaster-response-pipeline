# NLP and Machine Learning Pipeline for Disaster Response

## Classifying Social Media, SMS, and Hotline Messages to Deliver Urgent Information to Emergency Services, Including Medical and Search and Rescue Teams

### Overview
This project involves developing an ETL and machine learning pipeline designed to classify messages from various sources such as social media, SMS, and emergency hotlines. The goal is to categorize these messages into predefined emergency categories, ensuring that critical information quickly reaches the relevant response teams.

---

### 1. ETL Pipeline
In the ETL (Extract, Transform, Load) pipeline, the following processes are performed:
- **Preprocessing and Cleaning**: Columns of data, particularly the categories column, are cleaned and preprocessed.
- **Manual One-Hot Encoding**: Categorical variables are converted into a format suitable for machine learning.

### 2. Machine Learning Pipeline
In the machine learning pipeline, the following steps are implemented:
- **Tokenization Function**: A custom function is created to process text data.
- **NLP Pipelines with TF-IDF Vectorizer**: Various classifiers are employed:
  - **Random Forest Classifier**: Achieved an F1 score of **0.65**.
  - **Random Forest with TextLengthExtractor**: Yielded the same F1 score of **0.65**.
  - **Gradient Boosting Classifier**: Achieved an F1 score of **0.66**.
  - **Support Vector Machine (SVM)**: Performance results not specified.

After selecting the Gradient Boosting Classifier (GBM), a grid search was conducted to find the best hyperparameter combinations. Unfortunately, after adjusting the pipeline with the new hyperparameters, performance deteriorated to an F1 score of **0.57**.

### Experimentation with Undersampling
- Undersampling was performed using criteria proportional to the actual original count and the new undersampled data was shuffled.
- A pipeline comprising **TF-IDF Vectorizer**, **TextLengthExtractor**, and **Random Forest** was tested, resulting in a significant improvement with an F1 score of **0.78**.

---

### Why Excessive Hyperparameter Tuning Can Lead to Poorer Results
- **Overfitting**: Grid Search may identify hyperparameters that perform well on the training set but generalize poorly to the validation or test set.
- **Suboptimal Configuration**: Extensive tuning may explore hyperparameter spaces that lead to suboptimal configurations, despite appearing optimal within the grid.

### Importance of F1 Score in NLP, Especially with Imbalanced Data
The F1 score is crucial in this context as it provides a balanced measure of model performance, especially in imbalanced data scenarios. It accounts for both precision and recall, ensuring that the model performs well on minority classes, which are often critical in disaster response. Minimizing false positives and false negatives is essential in emergency contexts.

---

### Recommendations for Improvement
- **Use SMOTE Oversampling**:
  - Preserves valuable majority class data.
  - Generates synthetic data for minority classes, enhancing generalization.
  - Reduces the risk of overfitting compared to traditional oversampling.
  - Helps the model perform better in minority classes (e.g., critical emergencies) without sacrificing performance in majority classes.
  - Maintains or increases the overall dataset size, which is particularly beneficial for complex NLP tasks.

---

### Known Issues (Work in Progress)
- **Long Training Times**: The training process is significantly prolonged due to the nature of the data and the computational demands of the Random Forest classifier.

---

### Conclusion
This pipeline aims to enhance the effectiveness of emergency response systems by accurately classifying urgent messages. Continuous improvements, particularly in handling imbalanced data and optimizing model training times, are essential for achieving the desired outcomes.


C:\Anaconda3\disaster_project\Lib\site-packages\sklear
