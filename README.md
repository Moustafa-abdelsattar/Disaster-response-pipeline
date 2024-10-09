# NLP and Machine Learning Pipeline for Disaster Response
## Classifying Social Media, SMS, and Hotline Messages to Deliver Urgent Information to Emergency Services, Including Medical and Search and Rescue Teams

## Table of Contents
1. [Overview](#overview)
2. [ETL Pipeline](#etl-pipeline)
3. [Machine Learning Pipeline](#machine-learning-pipeline)
4. [Technical Terms](#technical-terms)
5. [Experimentation with Undersampling](#experimentation-with-undersampling)
6. [Why Excessive Hyperparameter Tuning Can Lead to Poorer Results](#why-excessive-hyperparameter-tuning-can-lead-to-poorer-results)
7. [Importance of F1 Score in NLP](#importance-of-f1-score-in-nlp-especially-with-imbalanced-data)
8. [Known Issues (Work in Progress)](#known-issues-work-in-progress)
9. [Recommendations for Improvement](#recommendations-for-improvement)
10. [Conclusion](#conclusion)
11. [Installation](#installation)

## 1. Overview
This project involves developing an ETL and machine learning pipeline designed to classify messages from various sources such as social media, SMS, and emergency hotlines. The goal is to categorize these messages into predefined emergency categories, ensuring that critical information quickly reaches the relevant response teams.

## 2. ETL Pipeline
In the ETL (Extract, Transform, Load) pipeline, the following processes are performed:
- **Preprocessing and Cleaning**: Columns of data, particularly the categories column, are cleaned and preprocessed.
- **Manual One-Hot Encoding**: Categorical variables are converted into a format suitable for machine learning.

## 3. Machine Learning Pipeline
In the machine learning pipeline, the following steps are implemented:
- **Tokenization Function**: A custom function is created to process text data.
- **NLP Pipelines with TF-IDF Vectorizer**: Various classifiers are employed:
  - **Random Forest Classifier**: Achieved an F1 score of 0.65.
  - **Random Forest with TextLengthExtractor**: Yielded the same F1 score of 0.65.
  - **Gradient Boosting Classifier**: Achieved an F1 score of 0.66.
  - **Support Vector Machine (SVM)**: Performance results not specified.

After selecting the Gradient Boosting Classifier (GBM), a grid search was conducted to find the best hyperparameter combinations. Unfortunately, after adjusting the pipeline with the new hyperparameters, performance deteriorated to an F1 score of 0.57.

## 4. Technical Terms
- **ETL**: Extract, Transform, Load
- **NLP**: Natural Language Processing
- **F1 Score**: A metric for evaluating the balance between precision and recall.

## 5. Experimentation with Undersampling
Undersampling was performed using criteria proportional to the actual original count, and the new undersampled data was shuffled. A pipeline comprising TF-IDF Vectorizer, TextLengthExtractor, and Random Forest was tested, resulting in a significant improvement with an F1 score of 0.78.

## 6. Why Excessive Hyperparameter Tuning Can Lead to Poorer Results
- **Overfitting**: Grid Search may identify hyperparameters that perform well on the training set but generalize poorly to the validation or test set.
- **Suboptimal Configuration**: Extensive tuning may explore hyperparameter spaces that lead to suboptimal configurations, despite appearing optimal within the grid.

## 7. Importance of F1 Score in NLP, Especially with Imbalanced Data
The F1 score is crucial in this context as it provides a balanced measure of model performance, especially in imbalanced data scenarios. It accounts for both precision and recall, ensuring that the model performs well on minority classes, which are often critical in disaster response. Minimizing false positives and false negatives is essential in emergency contexts.

## 8. Known Issues (Work in Progress)
Long Training Times: The Random Forest classifier demands significant computational resources, leading to prolonged training periods.

Data Quality and Representativeness: Model performance is heavily dependent on the quality of the training data. Poor representation of real-world scenarios can degrade accuracy.

Cultural and Linguistic Variations: Colloquialisms, slang, and language-specific nuances in messages can impact classification, necessitating a diverse training dataset.

Imbalanced Data Challenges: Despite using techniques like undersampling and SMOTE, the model may still struggle with classifying minority categories effectively.

Model Overfitting: There’s a risk of overfitting due to model complexity and training data volume, which can be mitigated through regularization and cross-validation.

Real-time Processing Limitations: The current setup may not be optimized for real-time message processing, affecting response times in urgent situations.

## 9. Recommendations for Improvement
- **Use SMOTE Oversampling**:
  - Preserves valuable majority class data.
  - Generates synthetic data for minority classes, enhancing generalization.
  - Reduces the risk of overfitting compared to traditional oversampling.
  - Helps the model perform better in minority classes (e.g., critical emergencies) without sacrificing performance in majority classes.
  - Maintains or increases the overall dataset size, which is particularly beneficial for complex NLP tasks.
 
-  Making a Python script to run the model from rather than simply a notebook to convey ideas and let it be actionable.

- **Real-time Monitoring and Updates**: Implement a monitoring system to evaluate model performance on new data over time. This can help identify when model retraining is necessary and ensure the system remains effective as language and communication patterns evolve.

- **User Interface Development**: Consider developing a user-friendly interface (e.g., a web app) for emergency responders to easily input messages and receive instant classifications, improving real-time decision-making.

## 10. Conclusion
This pipeline aims to enhance the effectiveness of emergency response systems by accurately classifying urgent messages. Continuous improvements, particularly in handling imbalanced data and optimizing model training times, are essential for achieving the desired outcomes.


## 11. Installation
To set up this project, clone the repository and install the required libraries:

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
pip install -r requirements.txt

