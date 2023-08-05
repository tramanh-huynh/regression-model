import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

data = pd.read_csv("StudentScore.csv")
data.info()
data.corr()
print(data['gender'].unique())

target = "math score"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ])

education_values = ["some high school","high school", "some college","associate's degree","bachelor's degree","master's degree"]
gender_values = ["female","male"]
lunch_values = ["standard","free/reduced"]
test_values = x_train["test preparation course"].unique()
ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', OrdinalEncoder(categories=[education_values,gender_values,lunch_values, test_values]))
    ])

nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', OneHotEncoder(sparse=False))
    ])

preprocessor = ColumnTransformer(transformers=[
    ("num_features", numerical_transformer, ["writing score","reading score"]),
    ("ord_features", ordinal_transformer, ["parental level of education","gender","lunch","test preparation course"]),
    ("nom_features", nominal_transformer, ["race/ethnicity"])
    ])


# model Random Forest Regression
reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
    ])

#model Linear Regression
# reg2 = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('model', LinearRegression())
#     ])

params = {
    "model__n_estimators": [50,100,200],
    "model__criterion": ["squared_error","absolute_error","friedman_mse","poisson"],
    "model__max_features": ["sqrt","log2", None],
    "preprocessor__num_features__imputer__strategy": ["median","mean"]
    }

grid_reg= GridSearchCV(reg, param_grid=params, cv=6, verbose=1, scoring="r2", n_jobs=-1)

#grid_reg = RandomizedSearchCV(reg, param_distributions=params, cv=6, verbose=1, scoring="r2", n_jobs=-1, n_iter=20)

grid_reg.fit(x_train, y_train)

y_predict = grid_reg.predict(x_test)
# for i,j in zip(y_predict, y_test):
#     print(i,j)

print("MSE: ", mean_squared_error(y_test, y_predict))
print("MAE: ", mean_absolute_error(y_test, y_predict))
print("R2: ", r2_score(y_test, y_predict))


# result = nominal_transformer.fit_transform(x_train[["race/ethnicity"]])
# for i, j in zip(x_train[["race/ethnicity"]].values, result):
#     print(i, j)












