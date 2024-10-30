import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
train_data = pd.read_csv('C:/Users/Jade Tong/Desktop/KAGGLE/home-data-for-ml-course/train.csv', index_col='Id')
test_data = pd.read_csv('C:/Users/Jade Tong/Desktop/KAGGLE/home-data-for-ml-course/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice              
train_data.drop(['SalePrice'], axis=1, inplace=True)

# Select numeric columns only
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
X = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()

#%%
#看下前几行的数据情况
X.head()

#%% 用pipeline来整理数据和建模，
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

#用n_estimators参数来设置trees的数目,用random_state来定随机种子
my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50, random_state=0))
                              ])

#%% 用scikit-learn中的vross_val_score来获取MAE，然后得到五个值的平均数，set the number of folds with the cv parameter
from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("Average MAE score:", scores.mean())
## Average MAE score: 18276.410356164386

#%% 用交叉验证来决定ml模型的参数，首先define一个‘get_score()'函数，输出3份folds的平均MAE
# the data in X and y to create folds,
# SimpleImputer() (with all parameters left as default) to replace missing values, and
# RandomForestRegressor() (with random_state=0) to fit a random forest model.

## n_estimators来表示随机森林的trees的数目，
def get_score(n_estimators):
    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators, random_state=0))
                              ])
    scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=3,
                              scoring='neg_mean_absolute_error')
    return scores.mean()

#%% 代入其它参数值来检验哪个模型最好，设置n_estimators为50,100,150,......,350,400
# 将结果储存在results(), where results[i] is the average MAE returned by get_score(i)
results = {}
for i in range(1,9):
    results[50*i] = get_score(50*i)
    
#%% visualize 上面的结果
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(list(results.keys()), list(results.values()))
plt.show()

# 从图上看trees为200时平均MAE最小












