# Energy and Housing Analysis Repository

This repository contains two primary projects: **Electricity Demand and Price Forecasting** and **S&P/Case-Shiller Home Price Index Analysis**. Below are the details for each project.

---

## 1. Electricity Demand and Price Forecasting

### 1. Introduction and Problem Statement
- The project focuses on forecasting electricity demand and price using time series data, consisting of two parts: energy consumption data and weather-related data.
- These variables are highly interconnected, as electricity demand and prices are influenced by external factors like temperature, humidity, and wind speed.
- The goal is to use machine learning models to forecast electricity prices and demand accurately, with applications in energy management, cost reduction, and grid stability.

### 2. Loading and Preparing the Data
- **Datasets**:
  - Energy dataset: Contains information on electricity prices, demand, and generation.
  - Weather dataset: Captures weather conditions like temperature, humidity, and wind speed.
- **Data Reading**: Loaded into the environment using pandas, with CSV files read for both datasets and preliminary analysis using `.head()` to inspect the first few rows.

### 3. Data Cleaning and Preprocessing
- **Handling Missing Values**: Missing or incomplete data is addressed using techniques like interpolation or forward fill to handle time series gaps.
- **Merging Datasets**: Energy and weather datasets are merged using a common key (likely date and time) to align corresponding records.
- **Feature Selection**: Relevant features include:
  - Energy dataset: Electricity prices, energy demand, energy generation.
  - Weather dataset: Temperature, humidity, wind speed, etc.

### 4. Feature Engineering and Normalization
- **Feature Engineering**:
  - Lag features: Energy demand from the previous day or hour to understand past patterns.
  - Rolling averages: Moving averages of energy consumption or weather variables to smooth fluctuations.
  - Time features: Day, month, hour, or season extracted from the datetime variable for periodic patterns.
- **Normalization/Scaling**: Min-Max scaling or Z-score standardization is applied to ensure similar feature ranges, enhancing model performance (e.g., neural networks, tree-based models).

### 5. Exploratory Data Analysis (EDA)
- **Visualizations**:
  - Line plots for trends in electricity prices and demand over time.
  - Scatter plots or heatmaps for relationships (e.g., temperature vs. demand, humidity vs. price).
  - Distribution plots for data spread and skewness.
- **Trend Analysis**: Identifies long-term trends, seasonality, or cyclic behaviors in energy demand data.
- **Correlation Analysis**: Assesses which weather factors (e.g., temperature, humidity) are most correlated with electricity demand and price changes.

### 6. Data Splitting
- **Train-Test Split**: Dataset divided into training and testing sets, with recent data (e.g., last few months) used for testing to preserve time order (no shuffling).

### 7. Model Selection and Training
- **Models Explored**:
  1. **Linear Regression**: Establishes a linear relationship but performed poorly due to complex, non-linear factors.
  2. **CATBoost**: Gradient boosting algorithm, excelling with categorical data and complex relationships, with higher R² and lower error rates than Linear Regression.
  3. **LightGBM**: Tree-based learning algorithm, known for speed and efficiency, outperforming others with the lowest MSE and RMSE.
- **Performance**: LightGBM captured complex interactions between demand, prices, and weather conditions most effectively.

### 8. Model Evaluation
- **Metrics**:
  - **Mean Squared Error (MSE)**: LightGBM had the lowest MSE, followed by CATBoost.
  - **Root Mean Squared Error (RMSE)**: LightGBM showed the smallest prediction error.
  - **R²**: LightGBM achieved the highest R² (0.954), explaining 95.4% of variance.
  - **Adjusted R²**: LightGBM’s high value confirmed no overfitting.

### 9. Final Model Comparison and Selection
- **Results**:
  - Linear Regression: Poor performance (high MSE, negative R²).
  - CATBoost: Better fit (R² 0.948), but outperformed by LightGBM.
  - LightGBM: Best performance (R² 0.954, MSE 9.17, RMSE 3.03), selected as the most accurate model.

### 10. Conclusion and Insights
- LightGBM emerged as the best model for forecasting electricity demand and prices.
- Advanced models like gradient boosting are crucial for capturing non-linear relationships in energy data.
- Practical applications include optimizing energy production, reducing costs, and planning for peak demand.

---

## 2. S&P/Case-Shiller Home Price Index Analysis

### 1. Objective
To analyze publicly available data on economic, demographic, and real estate indicators to build a predictive model that explains the impact of these factors on the S&P/Case-Shiller Home Price Index, a key indicator of U.S. home prices, over the last two decades.

### 2. Introduction
The S&P CoreLogic Case-Shiller Home Price Indices track the price levels of single-family homes in the United States. The S&P CoreLogic Case-Shiller U.S. National Home Price Index aggregates data from nine regions and 20 major metropolitan areas, updated monthly, measuring percentage changes while maintaining constant quality.

### 3. Data and Methodology
#### 3.1 Data Collection
Features identified via a literature survey of S&P CoreLogic Case-Shiller Home Price Indices, collected from [FRED](https://fred.stlouisfed.org/):
1. **UNRATE**: Unemployment Rate (Percent, Seasonally Adjusted, Monthly)
2. **CSUSHPISA**: S&P/Case-Shiller U.S. National Home Price Index (Index Jan 2000=100, Seasonally Adjusted, Monthly)
3. **PERMIT**: New Privately-Owned Housing Units Authorized: Total Units (Thousands of Units, Seasonally Adjusted Annual Rate, Monthly)
4. **PERMIT1**: New Privately-Owned Housing Units Authorized: Single-Family Units (Thousands of Units, Seasonally Adjusted Annual Rate, Monthly)
5. **MSACSR**: Monthly Supply of New Houses (Months' Supply, Seasonally Adjusted, Monthly)
6. **TTLCONS**: Total Construction Spending (Millions of Dollars, Seasonally Adjusted Annual Rate, Monthly)
7. **NASDAQCOM**: NASDAQ Composite Index (Index Feb 5, 1971=100, Not Seasonally Adjusted, Daily, Close)
8. **LFACTTTTUSM657S**: Active Population Growth Rate (Seasonally Adjusted, Monthly)
9. **HSN1F**: New One Family Houses Sold (Thousands, Seasonally Adjusted Annual Rate, Monthly)
10. **HOUST1F**: New Privately-Owned Housing Units Started: Single-Family Units (Thousands of Units, Seasonally Adjusted Annual Rate, Monthly)
11. **LFPR**: Labor Force Participation Rate [](https://pib.gov.in/PressReleaseIframePage.aspx?PRID=1966154)
12. **Housing Starts**: New Housing Project [](https://towardsdatascience.com/linear-regression-on-housing-csv-data-kaggle-10b0edc550ed)
13. **INDPRO**: Industrial Production: Cement [](https://fred.stlouisfed.org/series/INDPRO)
14. **Personal Income & Outlays**: CSV [](https://alfred.stlouisfed.org/release?rid=54)
15. **New Privately-Owned Housing Units Completed**: Total Units [](https://fred.stlouisfed.org/series/computsa)

#### 3.2 Data Preparation
Missing values in combined monthly and quarterly datasets are replaced with the mode value to ensure completeness and consistency.

#### 3.3 Exploratory Data Analysis
EDA uses ECDF (Empirical Cumulative Distribution Function) for histogram visualization and regression plotting to examine dataset variations.

#### 3.4 Model Selection and Evaluation
Models evaluated include linear regression, random forest, and XGBoost. Random forest performed best with an MSE of 8.33 and an adjusted R² of 0.99.

### 4. Results and Discussion
#### 4.1 Exploratory Data Analysis
HNFSEPUSSA follows the same trend as CSUSHPISA. LFPR, TTLCONS, and CPI Adjusted Price are positively correlated with HPI. UNRATE is inversely correlated with employment.

#### 4.2 Correlation Matrix
- New Privately-Owned Housing Units Started: Single-Family Units (0.94 with CSUSHPISA)
- Total Construction Spending (TTLCONS) (0.4 with CSUSHPISA)
- Monthly Supply of New Houses (0.84 with CSUSHPISA)
- New One Family Houses Sold (0.55 with CSUSHPISA)
- NASDAQ Composite Index (0.26 with CSUSHPISA)

#### 4.3 Machine Learning Models
Random forest was chosen for its balance of low MSE and high R² (0.997), effectively identifying important parameters.

### 5. Conclusions
- Key influencing factors on the S&P Home Price Index (HPI) include:
  - **Personal Income & Outlays**: Reflects changes in home prices.
  - **Employment rate**: Increases housing demand and prices.
  - **Total Construction Spending (TTLCONS)**: Affects housing supply and demand.
  - **New privately owned housing**: Measures property prices.
  - **CPI-Adjusted Price**: Reflects housing cost changes.
  - **NASDAQ Composite Index (NASDAQCOM)**: Influences economic growth and housing demand.
  - **Monthly Supply of New Houses (MSACSR)**: Impacts home prices based on supply relative to demand.

### MECE Framework
#### Factors Influencing US House Prices
- **Economic Factor**:
  - Growth in the Economy
  - Unemployment
  - Customer Trust Rates
  - Offering
  - GDP
  - Home Sales Economy Mirror
  - Supply and Demand
  - Advance
  - Compliance
  - Competition
  - Extinct
  - Surplus Productivity
- **Location**:
  - Neighborhoods
  - Highways
  - Attractions
  - Schools
  - Area Desirability
  - Crime Rate
- **Government**:
  - Government laws
  - Property Taxes
- **Banks**:
  - Mortgage Availability
  - Interest Rates

---

## 3. Heart Attack Prediction Model

### Introduction
This project aims to develop a predictive model for heart attack prediction using a dataset with various health-related features. The goal is to maximize recall and precision metrics instead of accuracy, focusing on correctly identifying cases with a higher chance of heart attack.

### Features
- **age**: Age of the patient
- **sex**: Sex of the patient
- **cp**: Chest pain type
  - 0 = typical angina
  - 1 = atypical angina
  - 2 = non-anginal pain
  - 3 = asymptomatic
- **trtbps**: Resting blood pressure in mm Hg
- **chol**: Cholesterol in mg/dl
- **exng**: Exercise induced angina
  - 1 = yes
  - 0 = no
- **fbs**: Fasting blood sugar > 120 mg/dl
  - 1 = true
  - 0 = false
- **restecg**: Resting electrocardiographic results
  - 0 = normal
  - 1 = having ST-T wave abnormality
  - 2 = showing probable or definite left ventricular hypertrophy
- **thalachh**: Maximum heart rate achieved
- **slp**: Slope
- **caa**: Number of major vessels
- **thall**: Thalium stress test result
- **target**:
  - 0 = less chance of heart attack
  - 1 = more chance of heart attack

### Assignment Objectives
1. **Data Exploration and Analysis**:
   - Explore the dataset and analyze relationships between features.
   - Identify correlations and visualize feature distributions.
2. **Data Pre-processing**:
   - Handle missing values, outliers, and address unbalanced data.
   - Perform feature engineering, including handling correlated features and scaling.
3. **Model Building**:
   - Split the dataset into training and testing sets.
   - Choose models like Logistic Regression, Random Forest, XGBoost.
4. **Model Evaluation**:
   - Evaluate using recall and precision metrics.
   - Visualize the confusion matrix.
5. **Presentation**:
   - Create a non-code report using Jupyter Notebook or export as PDF.
   - Summarize findings, insights, and visualizations.
6. **Explanation**:
   - Explain preprocessing steps, model selection, and metric choices.
   - Discuss the impact of certain features on heart attack prediction.

### Libraries/Package Used
1. `%matplotlib inline`: Enables inline plotting in Jupyter Notebook.
2. `numpy`: Fundamental library for numerical computing.
3. `pandas`: Powerful library for data manipulation and analysis.
4. `matplotlib.pyplot`: Comprehensive library for visualizations.
5. `seaborn`: High-level interface for statistical graphics.
6. `sklearn.model_selection`: Tools for dataset splitting.
7. `collections`: Contains the Counter class.
8. `sklearn.linear_model`: Provides LogisticRegression.
9. `sklearn.metrics`: Offers accuracy, precision, recall, F1-score, AUC-ROC.
10. `sklearn.ensemble`: Contains RandomForestClassifier.
11. `xgboost`: Library for gradient boosted decision trees.
12. `warnings`: Manages warning messages.
13. `warnings.simplefilter("ignore")`: Suppresses warning messages (use with caution).

### Conclusion
- Logistic Regression emerges as the most suitable model, given its high recall and reasonable precision.
- Correctly identifying individuals with a higher chance of heart attack is crucial.
- Further optimization of hyperparameters and feature engineering may enhance performance.
- Consider additional analyses to understand feature impact and identify improvement areas.

---

## Usage
- Clone the repository and use the provided data sources.
- Scripts for data processing and modeling are included for all projects.

## License
[MIT License](https://opensource.org/licenses/MIT)
