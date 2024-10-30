import pandas as pd
from sklearn.model_selection import train_test_split

# 读数
X = pd.read_csv('C:/Users/Jade Tong/Desktop/KAGGLE/home-data-for-ml-course/train.csv', index_col='Id')
X_test_full = pd.read_csv('C:/Users/Jade Tong/Desktop/KAGGLE/home-data-for-ml-course/test.csv', index_col='Id')

# 将带缺失值的行删掉，定变量
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice              
X.drop(['SalePrice'], axis=1, inplace=True)

# 分训练集和检验集
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# 选择分类变量中cardinality相对较低的变量 (convenient but arbitrary)
low_cardinality_cols=[cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# 选择数值变量
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# 合并
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)

#%% 建模1 Use the XGBRegressor class, and set the random seed to 0 (random_state=0). Leave all other parameters as default.
from xgboost import XGBRegressor
# Define the model
my_model_1 = XGBRegressor(random_state=0) # Your code here
# Fit the model
my_model_1.fit(X_train,y_train)

#%% 输出X_valid的预测值predictions_1，并算出与y_valid的MAE
# Get predictions
predictions_1 = my_model_1.predict(X_valid)

from sklearn.metrics import mean_absolute_error
# Calculate MAE
mae_1 = mean_absolute_error(y_valid,predictions_1) 

print("Mean Absolute Error:" , mae_1)
#Mean Absolute Error: 18161.82412510702

#%% 改善模型
# Begin by setting my_model_2 to an XGBoost model, using the XGBRegressor class. Use what you learned in the previous tutorial to figure out how to change the default parameters (like n_estimators and learning_rate) to get better results.
# Then, fit the model to the training data in X_train and y_train.
# Set predictions_2 to the model's predictions for the validation data. Recall that the validation features are stored in X_valid.
# Finally, use the mean_absolute_error() function to calculate the mean absolute error (MAE) corresponding to the predictions on the validation set. Recall that the labels for the validation data are stored in y_valid.

# Define the model
my_model_2 = XGBRegressor(n_estimators=1000,
                              learning_rate=0.03,
                              early_stopping_rounds=5) 

# Fit the model
my_model_2.fit(X_train,y_train,
               eval_set=[(X_valid, y_valid)], 
               verbose=False) 

# Get predictions
predictions_2 = my_model_2.predict(X_valid) 

# Calculate MAE
mae_2 = mean_absolute_error(y_valid,predictions_2) 

# Uncomment to print MAE
print("Mean Absolute Error:" , mae_2)
# Mean Absolute Error: 16959.117950021406















