#### Gradient Boosting 梯度提升算法，使误差梯度下降'gradient descent'
#### 开始与建立一个简单模型，得出估计值后进入一个环形程序
## First, we use the current ensemble to generate predictions for each observation in the dataset. To make a prediction, we add the predictions from all models in the ensemble.
## These predictions are used to calculate a loss function (like mean squared error, for instance).
## Then, we use the loss function to fit a new model that will be added to the ensemble. Specifically, we determine model parameters so that adding this new model to the ensemble will reduce the loss. (Side note: The "gradient" in "gradient boosting" refers to the fact that we'll use gradient descent on the loss function to determine the parameters in this new model.)
## Finally, we add the new model to ensemble, and ...
## ... repeat!

#####教程代码
#%% 1.load the training and validation data
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv('C:/Users/Jade Tong/Desktop/KAGGLE/Melbourne Housing Snapshot/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

#%% XGBoost---Extreme Gradient Boosting 是梯度提升的一种方法，专注于速度和精准度
from xgboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(X_train, y_train)

#%% 预测并测量模型
from sklearn.metrics import mean_absolute_error

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
## Mean Absolute Error: 238515.2459125322

#%% 改进参数1
#  XGBoost里面有几个参数是可以显著影响准确率和训练速度的，一个是树的数量(n_estimators)
#  n_estimators说明模型进入环形检验的次数，太少会导致underfitting，太多会overfitting（对训练data有效但检验data无效）
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train)

#%% 改进参数2 early_stopping_rounds
#  early_stopping_rounds可以自动找到n_estimators的最优值，
#  Early stopping causes the model to stop iterating when the validation score stops improving
#  一般设置一个较高的n_estimators，加early_stopping_rounds去找最优值
my_model = XGBRegressor(n_estimators=500,early_stopping_rounds=5)
my_model.fit(X_train, y_train,  
             eval_set=[(X_valid, y_valid)],   # 用early_stopping_rounds时，要注明哪些数据是用来检验的
             verbose=False)

#%% 改进参数3 learning_rate 学习率，控制模型的学习进度
#  In general, a small learning rate and large number of estimators will yield more accurate XGBoost models
my_model = XGBRegressor(n_estimators=1000,early_stopping_rounds=5,learning_rate=0.05)  ##默认learning_rate为0.1
my_model.fit(X_train, y_train, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)

#%% 改进参数4 n_jobs 用于大数据 一般设置n_jobs为电脑CPU内核数
my_model = XGBRegressor(n_estimators=1000,early_stopping_rounds=5, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train, y_train,
             eval_set=[(X_valid, y_valid)], 
             verbose=False)





