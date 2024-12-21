import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, LabelEncoder, FunctionTransformer
from feature_engine.outliers import Winsorizer
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_excel(r"C:\Users\raksh\Downloads\Rotten_Tomatoes_Movies3.xls")
dataset.info()

# Dataframe
df1 = dataset.drop_duplicates()
df1 = df1.dropna(subset = 'audience_rating')
df1.info()

df2 = df1.dropna(subset = 'runtime_in_minutes')
df2 = df2[['rating', 'genre', 'directors', 'runtime_in_minutes', 'studio_name',
           'tomatometer_status', 'tomatometer_rating', 'tomatometer_count', 'audience_rating']]
df2_description = df2.describe()

# Feature names
column_names = ['rating', 'genre', 'directors', 'runtime_in_minutes', 'studio_name',
                'tomatometer_status', 'tomatometer_rating', 'tomatometer_count']
columns_to_be_encoded = ['rating', 'genre', 'directors', 'studio_name', 'tomatometer_status']
numeric_columns = ['runtime_in_minutes', 'tomatometer_rating', 'tomatometer_count']

# Label Encoding
def label_encoding(df, columns_to_be_encoded):
    label_enc = LabelEncoder()
    df[columns_to_be_encoded] = df[columns_to_be_encoded].astype('str')
    for column in df[columns_to_be_encoded]:
        df[column] = label_enc.fit_transform(df[column])
    return df

# Outlier treatment
wins = Winsorizer(capping_method = 'iqr',
                  tail = 'both',
                  fold = 1.5,
                  variables = column_names)

# Preprocessing pipeline
preprocess_pipe = Pipeline(steps = [('label_encode', FunctionTransformer(lambda x: label_encoding(x, columns_to_be_encoded), validate=False)),
                             ('outlier', ColumnTransformer(transformers = [('iqr', wins, column_names)])),
                             ('imputation', SimpleImputer(strategy = 'median')),
                             ('log_transform', PowerTransformer(method='yeo-johnson')),
                             ('scaling', MinMaxScaler())])

preprocessor = ColumnTransformer(transformers = [('numerical', preprocess_pipe, column_names)])

# Train-Test Split
X = df2.drop(columns = 'audience_rating')
y = df2['audience_rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 20)

# Model Building
xgb_model = Pipeline(steps=[('preprocess', preprocessor),
                            ('regress', XGBRegressor(random_state=20))])

xgb_model.fit(X_train, y_train)

# Step 6: Model Validation
test_pred = xgb_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_r2 = r2_score(y_test, test_pred)
train_pred = xgb_model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
train_r2 = r2_score(y_train, train_pred)

print(f'Train RMSE: {train_rmse}')
print(f'Train R²: {train_r2}')
print(f'Test RMSE: {test_rmse}')
print(f'Test R²: {test_r2}')


# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_dist = {'regress__n_estimators': randint(100, 500),
              'regress__learning_rate': uniform(0.01, 0.2),
              'regress__max_depth': randint(3, 15),
              'regress__subsample': uniform(0.6, 0.4),
              'regress__colsample_bytree': uniform(0.6, 0.4),
              'regress__gamma': uniform(0, 0.3)}

# RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(estimator = xgb_model, param_distributions = param_dist, 
                                   n_iter = 50, cv = 3, n_jobs = -1, verbose = 2, 
                                   scoring='neg_mean_squared_error', random_state = 20)

random_search.fit(X_train, y_train)

# Best parameters from RandomizedSearchCV
print("Best Parameters Found: ", random_search.best_params_)

# Get the best model and evaluate
best_xgb_model_random = random_search.best_estimator_
train_pred_random = best_xgb_model_random.predict(X_train)
train_rmse_random = np.sqrt(mean_squared_error(y_train, train_pred_random))
train_r2_random = r2_score(y_train, train_pred_random)
test_pred_random = best_xgb_model_random.predict(X_test)
test_rmse_random = np.sqrt(mean_squared_error(y_test, test_pred_random))
test_r2_random = r2_score(y_test, test_pred_random)

print(f'Train RMSE (RandomizedSearchCV): {train_rmse_random}')
print(f'Train R² (RandomizedSearchCV): {train_r2_random}')
print(f'Test RMSE (RandomizedSearchCV): {test_rmse_random}')
print(f'Test R² (RandomizedSearchCV): {test_r2_random}')



# Model pipeline with XGBRegressor and the best hyperparameters
final_pipeline = Pipeline(steps=[('preprocess', preprocessor),
                                 ('regress', XGBRegressor(
                                     n_estimators=307,  # Best hyperparameter found from RandomizedSearchCV
                                     learning_rate=0.0269,
                                     max_depth=3,
                                     subsample=0.6038,
                                     colsample_bytree=0.9227,
                                     gamma=0.1065,
                                     random_state=20))])


# Predict using this final pipeline:
final_pipeline.fit(X_train, y_train)
final_train_pred = final_pipeline.predict(X_train)
final_test_pred = final_pipeline.predict(X_test)

final_train_rmse = np.sqrt(mean_squared_error(y_train, final_train_pred))
final_train_r2 = r2_score(y_train, final_train_pred)
final_test_rmse = np.sqrt(mean_squared_error(y_test, final_test_pred))
final_test_r2 = r2_score(y_test, final_test_pred)

print(f'Final Train RMSE: {final_train_rmse}')
print(f'Final Train R²: {final_train_r2}')
print(f'Final Test RMSE: {final_test_rmse}')
print(f'Final Test R²: {final_test_r2}')











