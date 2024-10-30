#交叉验证，将数据集分成5份（5 folds），每一份fold（20%）作为valid建一次模，
# 实验1, 用第一份fold做validation(or holdout) 将其他数据为training data.
# 实验2, 用第二份fold做validation，如此类推
# 数据集量小，要用交叉验证； 数据集量大，单一验证就足够了

######教程代码
#%% 1. 定好变量
import pandas as pd

# Read the data
data = pd.read_csv('C:/Users/Jade Tong/Desktop/KAGGLE/Melbourne Housing Snapshot/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

#%% 2. 用pipeline来填充缺失值和建一个随机森林模型来做预测
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50,random_state=0))
                             ])

#%% 3. 用scikit-learn包中的cross_val_score()来得出 ‘cross-validation score’，函数中用‘cv’来设置folds的份数
from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE  
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)  ###输出5次建模得出的平均绝对误差MAE

##一般用5个模型的平均MAE来表示，so
print("Average MAE score (across experiments):")
print(scores.mean())
