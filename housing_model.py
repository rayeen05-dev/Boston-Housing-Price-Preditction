import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error
columns = [
"crime_rate",
"large_lot_residential_ratio",
"non_retail_business_ratio",
"near_charles_river",
"nitric_oxide_level",
"avg_rooms_per_house",
"old_house_ratio",
"distance_to_job_centers",
"highway_access_index",
"property_tax_rate",
"student_teacher_ratio",
"racial_demographic_index",
"low_income_population_pct",
"median_house_value"
]

df = pd.read_csv("housing.csv", header=None, names=columns)

df["rooms_per_low_income"] = df["avg_rooms_per_house"] / (df["low_income_population_pct"] + 1)

"""def split_train_test(data,test_ratio) :
    index = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_index = index[:test_set_size]
    train_index = index[test_set_size:]
    return data.iloc[train_index] , data.iloc[test_index]"""


train_set, test_set = train_test_split(df,test_size=0.3,random_state=42   ) 
#random_state for having the same split every time u run the model

# performing a stratified split 

df["income_cat"] = np.ceil(df["median_house_value"] / 10)
df["income_cat"] = df["income_cat"].clip(upper=5)

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index , test_index in split.split(df,df["income_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

# comparing propotions 

"""def income_cat_propotions(data):
    return data["income_cat"].value_counts(normalize=True)

compare_props = pd.DataFrame({
    "overall" : income_cat_propotions(df),
    "train" : income_cat_propotions(strat_train_set),
    "test" : income_cat_propotions(strat_test_set)


})

print(compare_props)"""

#remove income_cat after stratifing the data 

for set_ in (strat_test_set , strat_train_set):
    set_.drop("income_cat",axis=1,inplace=True)


df = df.drop("income_cat",axis=1)
#which variables affect house prices the most

#corr_matrix = df.corr()
#print(corr_matrix["median_house_value"].sort_values(ascending=False))

#attributs = ["median_house_value","avg_rooms_per_house","student_teacher_ratio","low_income_population_pct"]
#scatter_matrix(df[attributs],figsize=(12,8))
#plt.show()

#reverting to a clean training data set 
df = strat_train_set.drop("median_house_value" , axis=1) #reseting to a clean data set
df_labels = strat_train_set["median_house_value"].copy() #storing the target sepratly 


#a pipeline with an imputer and a scaler 


pipeline = Pipeline([
    ("imputer" , SimpleImputer(strategy="mean")),
    ("scaler",StandardScaler())

])
x_prepared = pipeline.fit_transform(df)


df_scaled = pd.DataFrame(x_prepared,columns=df.columns)

#print(df_scaled.describe())

#training the linear regression model 

regressor  = RandomForestRegressor()
gs = GridSearchCV(cv=5 , error_score=np.nan,estimator=regressor,
                  param_grid={
                      'max_depth' : [None,5,10,20],
                      'n_estimators' : [100,250,700]

                  })

gs.fit(df_scaled,df_labels)


regressor.fit(df_scaled,df_labels)

#prepare test set 
x_test = strat_test_set.drop("median_house_value",axis=1)
y_test = strat_test_set["median_house_value"].copy()
x_test_scaled = pd.DataFrame(
    pipeline.transform(x_test) , 
    columns=x_test.columns,
    index=x_test.index
)
#choosing best model
best_rf = gs.best_estimator_
predictions = best_rf.predict(x_test_scaled)
print(predictions)
#comparing 
comp = pd.DataFrame({
    "prediction":predictions,
    "actual":y_test

})
print(comp.head(10))
rmse = root_mean_squared_error(y_test,predictions)
print(f"RMSE : {rmse}")
