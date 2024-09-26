######################### Libraries #########################

import pyarrow.parquet as pq
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB  # Naive Bayes is not typically used for regression, consider removing this model
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
import joblib

import time
import logging
import os
from datetime import datetime

######################### Configuration #########################

output_table = 'rfm_test'

logger = logging.getLogger('RFM_Prediction_logger')
# Set the level for the logger
logger.setLevel(logging.INFO)
LOGS_DIR = 'logs'
os.makedirs(LOGS_DIR, exist_ok=True)

# Randomness for reproductivity
SEED = 7
np.random.seed(SEED)

# n = Number of predictions to made , ex: 18 means the next 6 months
n = 18 
count = n - 11 

dfs = {}
rfm_predictions = {}

def create_log_file(filename):
    # Create handlers
    global logger
    current_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    # log_filename = os.path.join(log_dir, f'yamuna_{current_time}.log')
    log_filename = os.path.join(LOGS_DIR, f'{filename}_{current_time}.log')

    file_handler = logging.FileHandler(log_filename)

    # Set levels for handlers
    file_handler.setLevel(logging.INFO)

    # Create formatters and add them to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)

create_log_file(output_table)
######################### Importing datasets #########################

logger.info('Importing datasets starts')
start_time = time.time()
rfm = pd.read_parquet('customer_rfm_demo_product.parquet', engine='pyarrow')
product = pd.read_parquet('product_gap.parquet', engine='pyarrow')
#print("--- %s seconds for importing datasets ---" % (time.time() - start_time))
#print("Done for importing datasets")
logger.info(f'Importing datasets ends - Time taken: {time.time() - start_time}')

######################### Pre-processing #########################

logger.info('Preprocessing starts')
start_time = time.time()
# Merging two dataset 
temp = pd.merge(rfm, product, left_on ='cid', right_on = 'cid', how = 'inner')

# Selecting rows based on condition 
filtered = temp[temp['pairing_gap'] == 1] 

# Drop repeated columns
df_dropped = filtered.drop(['age_y', 'city_y', 'disability_status_y', 'dob_y',
                            'education_y', 'gender_y', 'income_level_y', 'income_range_million_idr_y',
                            'marital_status_y', 'nationality_y', 'no_dependents_y', 
                            'occupation_y', 'phone_number_y', 'postcode_y', 
                            'race_y', 'religion_y', 'state_y'], axis=1)

# Convert numeric month to abbreviated month names (Jan, Feb, Mar, etc.)
df_dropped['month_name'] = pd.to_datetime(df_dropped['transaction_month'], format='%m').dt.strftime('%b')

# Combine 'month' and 'year' into one column
df_dropped['month_year'] = df_dropped['month_name'].astype(str) + ' ' + df_dropped['transaction_year'].astype(str)

# Drop repeated columns
final = df_dropped.drop(['transaction_month', 'transaction_year', 'month_name',  'dob_x'], axis=1)

# Preparation for Label Encoder
categorical_list = ['customer_segment','month_year']

lab_encoder = LabelEncoder()

for item in categorical_list:
    try : 
        final[item] = lab_encoder.fit_transform(final[item])
    except : 
        print("Fail to encode in : ", item)

# Move column 'A' to the end
df = final[[col for col in final.columns if col != 'rfm_score'] + ['rfm_score']]

df_feature_imp=df[['cid','customer_segment','monetary_score','frequency_score','month_year','rfm_score']]
df_feature_imp_name = df_feature_imp.columns

# Train and test splitting 
X = df_feature_imp[df_feature_imp_name[1:-1]]
Y = df_feature_imp[df_feature_imp_name[-1]]
X_train, X_test, y_train, y_test =train_test_split(X,Y,
                                                   test_size=0.25,
                                                   random_state=0)
#print("--- %s seconds for data preprocessing ---" % (time.time() - start_time))
#print("Done for data preprocessing")
logger.info(f'Preprocessing ends - Time taken: {time.time() - start_time}')
######################### Base Modelling #########################

logger.info('Base modelling starts')
start_time = time.time()
def GetBasedModel2():
    basedModels = []
    basedModels.append(('CART', DecisionTreeRegressor()))
    #basedModels.append(('SVM', SVR()))
    basedModels.append(('GBM', GradientBoostingRegressor()))
    basedModels.append(('RF', RandomForestRegressor()))
    return basedModels

def BasedLine3(X_train, y_train, models):
    # Evaluation metric
    scoring = 'neg_mean_squared_error'  # Use MSE for regression

    results = []
    names = []
    for name, model in models:
        start_time = time.time()
        # Train the model on the entire training dataset
        model.fit(X_train, y_train)
        
        # Evaluate the model using MSE on the training data
        y_pred = model.predict(X_train)
        mse = mean_squared_error(y_train, y_pred)
        
        results.append(mse)
        names.append(name)
        
        msg = "--- %s: %f MSE ---" % (name, mse)
        #print(msg)
        #print("--- %s seconds ---" % (time.time() - start_time))
        #print()

    return names, results

def TestModels(X_test, y_test, models):
    # Evaluation metric
    scoring = 'neg_mean_squared_error'  # Use MSE for regression

    results = []
    names = []
    for name, model in models:
        start_time = time.time()
        
        # Predict on the test dataset
        y_pred = model.predict(X_test)
        
        # Evaluate the model using MSE on the test data
        mse = mean_squared_error(y_test, y_pred)
        
        results.append(mse)
        names.append(name)
        
        msg = "--- %s: %f MSE ---" % (name, mse)
        #print(msg)
        #print("--- %s seconds ---" % (time.time() - start_time))
        #print()

    return names, results

models = GetBasedModel2()
train_model,train_model = BasedLine3(X_train, y_train,models)
test_model,test_result = TestModels(X_test, y_test,models)
#print("--- %s seconds for base modelling ---" % (time.time() - start_time))
#print("Done for base modelling")
logger.info(f'Base modelling ends - Time taken: {time.time() - start_time}')
######################### Final Modelling #########################

logger.info('Final modelling starts')
start_time = time.time()
model = RandomForestRegressor()
model.fit(X_train, y_train)

def save_model(model, filename):
    # Save the trained model to a file
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

save_model(model, 'RF_Regressor')

def load_model(filename):
    # Load the model from the file
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

model = load_model('RF_Regressor')
#print("--- %s seconds for final modelling ---" % (time.time() - start_time))
#print("Done for final modelling")
logger.info(f'Final modelling ends - Time taken: {time.time() - start_time}')
######################### Prediction Data Processing #########################

logger.info('Prediction data processing starts')
start_time = time.time()
# Filter rows where 'month_year' is equal to 11
filtered_df = df_feature_imp[df_feature_imp['month_year'] == 11]

cleaned_df = filtered_df.drop_duplicates(subset='cid', keep='last')

def data_processing(data: pd.DataFrame, n:int) -> None:

    #cleaned_df['month_year'] = cleaned_df['month_year'].replace(11, 12)
    data.loc[:,'month_year'] = n   
    return data

#print("Len of cleaned data : ", len(cleaned_df))
df = cleaned_df.copy()

for i in range(12, n):
    temp_df = cleaned_df.copy()  # Create a copy of cleaned_df to avoid overwriting
    temp_df['month_year'] = i  # Assign the correct month_year value
    dfs[f'df{i-11}'] = temp_df  # Store the modified DataFrame in the dictionary

def batch_predict(column_name, model, X_batch):
    # Use the loaded model to predict on new batch data
    rfm_predicted = [] 
    X_batch=X_batch[['customer_segment','monetary_score','frequency_score','month_year']]
    # Iterate through each row in the DataFrame and make predictions
    for index, row in X_batch.iterrows():
        #print("Row : ", type(row))
        #row = row.drop(labels=['cid'])
        result = model.predict(row.to_frame().T)
        rfm_predicted.append(result.item())
        #print(row)

    return rfm_predicted

count = n - 11
for i in range(1, count):
    # Perform batch prediction and immediately assign the result to the DataFrame
    df = df.assign(**{f'rfm_pm{i}': batch_predict(f'rfm_p{i}', model, dfs[f'df{i}'])})

#print("--- %s seconds for prediction ---" % (time.time() - start_time))
#print("Done for prediction")
logger.info(f'Prediction data processing ends - Time taken: {time.time() - start_time}')

logger.info('Result saving starts')
start_time = time.time()
# Save DataFrame to a Parquet file
df.to_parquet('predicted_rfm.parquet', engine='pyarrow', index=False)
logger.info(f'Result saving ends - Time taken: {time.time() - start_time}')
#print("Done for saving result into parquet file")