# Boston Housing Price Prediction

This project is a **machine learning pipeline** to predict **median house values** using a subset of housing data. It demonstrates a complete ML workflow including **data preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation**.  

The goal is to predict housing prices in a given area using several features such as crime rate, number of rooms, proximity to highways, and low-income population percentage.  

---

## Dataset

The dataset used contains the following columns:

| Column | Description |
|--------|-------------|
| crime_rate | Crime rate per town |
| large_lot_residential_ratio | Proportion of large-lot residential land |
| non_retail_business_ratio | Ratio of non-retail business acres |
| near_charles_river | 1 if tract bounds river, 0 otherwise |
| nitric_oxide_level | Concentration of nitric oxide |
| avg_rooms_per_house | Average number of rooms per house |
| old_house_ratio | Proportion of owner-occupied units built before 1940 |
| distance_to_job_centers | Weighted distance to five employment centers |
| highway_access_index | Index of highway accessibility |
| property_tax_rate | Property tax rate per $10,000 |
| student_teacher_ratio | Number of students per teacher |
| racial_demographic_index | Proportion of certain demographic groups |
| low_income_population_pct | Percent of low-income population |
| median_house_value | Target variable: median value of houses |
| rooms_per_low_income | Engineered feature: `avg_rooms_per_house / (low_income_population_pct + 1)` |

> **Note:** The last column is added as a feature to improve predictive power.

---

## Workflow

The project follows a **typical machine learning pipeline**:

### 1. Data Splitting

- The dataset is split into **training** (70%) and **testing** (30%) sets.  
- Additionally, a **stratified split** is performed based on median house value categories to ensure representative samples.

### 2. Feature Engineering

- A new feature is created:
rooms_per_low_income = avg_rooms_per_house / (low_income_population_pct + 1)

### 3. Data Cleaning and Scaling 
 -missing values were imputed using mean strategy(simpleImputer)
 -Features are scaled with StandardScaler for models sensitive to feature magnitude.
### 4. Model Training 
  Random Forest Regressor is used as the primary model.

  GridSearchCV with 5-fold cross-validation is used to tune hyperparameters:
          max_depth: [None, 5, 10, 20]
          n_estimators: [100, 250, 700]
### 5.Evaluation 
  Predictions are made on the test set.

  Results are compared with actual values:
           prediction  actual
503   28.18    23.9
83    23.65    22.9
9     19.51    18.9
417    8.64    10.4
213   23.15    28.1

