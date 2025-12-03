# Project Proposal: User Story 6 - Traffic Prediction & Urban Mobility Analysis

**Team Leader:** Omar Khaled

**Team Members:** 
1. Omar Khaled (Team Leader)
2. Elias Zerz
3. Ahmed Omran
4. Arsany Milad
5. Kisho

**Project Deadline:** December 4, 2025

**Date:** November 23, 2025

---

## 1. Introduction and Problem Definition

Modern urban environments are increasingly reliant on ride-sharing services like Uber and Lyft for daily transportation. These services generate a vast amount of data that holds the key to understanding and optimizing urban mobility. The core challenge of this project is to harness this data to build a robust machine learning system capable of predicting ride prices with high accuracy while simultaneously uncovering deeper patterns in transportation demand and driver-rider behavior. By analyzing historical ride data from Boston, MA, we aim to develop a comprehensive solution that not only provides predictive insights but also offers a descriptive analysis of mobility trends. This will enable better decision-making for urban planners, transportation services, and consumers.

The project directly addresses **User Story 6: Traffic Prediction & Urban Mobility Analysis**, which requires a multi-faceted approach involving data preprocessing, exploratory analysis, supervised and unsupervised machine learning, deep learning, and the deployment of an interactive dashboard.

---

## 2. Dataset Specification

To address the problem, we will utilize the **"Uber and Lyft Dataset Boston, MA"** available on Kaggle [1]. This dataset provides a rich source of information for building and validating our models.

| **Attribute** | **Description** |
| :--- | :--- |
| **Dataset Name** | Uber and Lyft Dataset Boston, MA |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma) |
| **Total Records** | ~693,000 rides |
| **Project Subset** | 100,000 rides (for manageable processing) |
| **Time Period** | Nov 26, 2018 – Dec 18, 2018 |
| **Features** | 57 columns, including temporal, spatial, weather, and service-specific data |
| **Target Variable** | `price` (Continuous value for regression) |

Key features include timestamps, source/destination locations, cab type (Uber/Lyft), and corresponding weather data for each ride, making it an ideal choice for this user story.

---

## 3. Proposed ML Pipeline and Team Responsibilities

The project will be executed in five distinct phases, with each phase having a dedicated lead responsible for its successful completion. All team members will collaborate throughout the project, but clear ownership will ensure all mandatory components are integrated seamlessly.

### Phase 1: Data Preprocessing & Exploratory Data Analysis (EDA)

This foundational phase is crucial for ensuring data quality and uncovering initial insights that will guide the modeling stages.

**Team Member: Elias Zerz**

**Responsibilities:**
- Lead the complete data preprocessing pipeline including data loading, cleaning, and initial exploration
- Handle missing values using appropriate imputation techniques and document all data quality issues
- Perform geospatial processing of pickup and dropoff coordinates and calculate distance features using the Haversine formula
- Engineer comprehensive temporal features including rush hour indicators, holiday effects, time-of-day categories, and weekend flags
- Handle high-cardinality categorical variables (location pairs) using target encoding and other advanced encoding techniques
- Address pricing outliers and surge multiplier effects while preserving important patterns
- Apply feature scaling and normalization using RobustScaler for numerical features
- Create temporal train-test splits to prevent data leakage
- Conduct comprehensive Exploratory Data Analysis including geographic heat maps of ride density and pricing variations
- Analyze temporal patterns by hour, day, and season to identify peak demand periods
- Compare Uber vs Lyft services in terms of pricing, availability, and market share
- Identify surge pricing triggers and analyze correlation with external factors (weather, time, demand)
- Perform route popularity analysis and congestion correlation studies
- Generate comprehensive visualizations and an EDA report documenting all findings

---

### Phase 2: Supervised Machine Learning (Regression)

This phase focuses on building predictive models to estimate ride prices based on the features engineered in Phase 1.

**Team Member: Omar Khaled**

**Responsibilities:**
- Lead the development and evaluation of supervised regression models for price prediction
- Implement and tune XGBoost regressor with spatial-temporal cross-validation and hyperparameter optimization
- Build Random Forest regressor with feature importance analysis for business insights
- Train Gradient Boosting regressor with categorical feature support and early stopping mechanisms
- Develop baseline Linear Regression models with Ridge/Lasso regularization for comparison
- Perform comprehensive feature engineering for spatial interactions and temporal effects
- Evaluate all models using multiple metrics including RMSE, MAE, R² score, quantile loss, and MAPE
- Conduct spatial-temporal cross-validation to ensure robust model performance
- Generate SHAP analysis plots for model interpretability and feature importance rankings
- Create detailed model comparison report with performance analysis by route, time, and service type
- Document all modeling decisions and provide recommendations for model deployment

---

### Phase 3: Unsupervised Machine Learning

This phase focuses on discovering hidden structures and patterns within the ride data to segment mobility behaviors.

**Team Member: Ahmed Omran**

**Responsibilities:**
- Lead all unsupervised learning tasks including clustering and dimensionality reduction
- Apply KMeans clustering (k=10) on spatio-temporal ride patterns to identify distinct mobility segments
- Use DBSCAN for anomaly detection in pricing and routing to identify unusual patterns
- Implement hierarchical clustering for nested pattern discovery and multi-level segmentation
- Perform dimensionality reduction using PCA to reduce feature space and explain variance
- Apply t-SNE and UMAP for 2D/3D visualization of high-dimensional ride patterns
- Validate clusters using silhouette score, Davies-Bouldin index, and Calinski-Harabasz score
- Create comprehensive cluster profiles with statistical summaries, temporal characteristics, spatial patterns, and price distributions
- Validate clusters with urban planning characteristics to ensure meaningful segmentation
- Generate cluster assignment for all rides and create detailed cluster profile reports
- Produce dimensionality reduction visualizations and anomaly detection results

---

### Phase 4: Deep Learning

This phase involves building advanced neural network architectures to capture complex, non-linear relationships in the data.

**Team Member: Arsany Milad**

**Responsibilities:**
- Lead the development of all deep learning models for the project
- Build and train a Deep Neural Network (DNN) baseline with multiple hidden layers, dropout regularization, and batch normalization
- Implement LSTM (Long Short-Term Memory) networks for temporal price pattern prediction and time-series forecasting
- Develop Autoencoder architecture for unsupervised ride pattern representation learning and anomaly detection
- Apply attention mechanisms to LSTM models for interpretable predictions with attention weight visualization
- Compare deep learning models against traditional machine learning methods (XGBoost, Random Forest)
- Evaluate LSTM performance against statistical time-series models (ARIMA, Prophet)
- Optimize model architectures through hyperparameter tuning and regularization techniques
- Generate performance comparison reports with metrics including RMSE, MAE, and R² scores
- Create attention visualization plots and learned embeddings from autoencoder
- Document model architecture diagrams and training procedures
- Analyze the trade-off between interpretability and accuracy for all deep learning models

---

### Phase 5: Streamlit Deployment

The final phase involves creating an interactive web application to present the project's findings and models to end-users.

**Team Member: Kisho**

**Responsibilities:**
- Lead the design and deployment of the comprehensive Streamlit dashboard application
- Create a real-time price prediction interface with input forms for route, time, date, and cab type
- Develop a mobility pattern explorer with interactive filters for time range, location, and cab type
- Build a surge pricing alert system with historical pattern analysis and trend forecasting
- Design a service comparison dashboard showing Uber vs Lyft performance metrics and pricing
- Implement data export functionality for urban planning and policy development recommendations
- Integrate all pre-trained models (supervised, unsupervised, and deep learning) into the application
- Create interactive visualizations using Plotly and Folium for geographic mapping
- Implement SHAP force plots for model explainability in the prediction interface
- Ensure responsive design and optimize application performance with data caching
- Develop comprehensive user documentation and deployment instructions
- Test the application thoroughly to ensure response time under 2 seconds
- Deploy the application on Streamlit Cloud or local server with proper environment management

---

## 4. Project Integration and Timeline

All phases are interconnected and will require continuous collaboration. The output of the **Data Preprocessing** phase is the input for all subsequent modeling and analysis phases. Insights from **EDA** will inform feature engineering for the **Supervised ML** models. The clusters identified in the **Unsupervised ML** phase can be used as features in the **Deep Learning** models. Finally, the best-performing models from all analytical phases will be integrated into the **Streamlit Deployment**.

Given the project deadline of **December 4, 2025**, the following timeline is proposed:

| **Week** | **Dates** | **Phase** | **Lead** | **Deliverables** |
| :--- | :--- | :--- | :--- | :--- |
| Week 1 | Nov 23 - Nov 29 | Phase 1: Data Preprocessing & EDA | Omar Khaled | Cleaned dataset, EDA report, visualizations |
| Week 2 | Nov 30 - Dec 4 | Phase 2: Supervised ML | Elias Zerz | Trained models, performance report |
| Week 2 | Nov 30 - Dec 4 | Phase 3: Unsupervised ML | Ahmed Omran | Cluster analysis, dimensionality reduction |
| Week 2 | Nov 30 - Dec 4 | Phase 4: Deep Learning | Arsany Milad | DNN, LSTM, Autoencoder models |
| Week 2 | Nov 30 - Dec 4 | Phase 5: Streamlit Deployment | Kisho | Interactive dashboard, documentation |

**Note:** Due to the tight deadline, Phases 2-5 will run in parallel with coordination from the team leader. Phase 1 must be completed first to provide clean data for all other phases.

---

## 5. References

[1] B. L. L. R. B. (2018). *Uber and Lyft Dataset Boston, MA*. Kaggle. Retrieved from https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma
