# 当你的训练数据集含有关于目标的信息，但是下次用来预测的时候，不一定有相同的情况
# 这样会导致在训练集上拟合得好，但是在在真实的世界中做决策的时候，模型变得非常不准确
# 有两大类leakage：target leakage和train—test contamination

#%%  1. Target Leakage 当模型中带有预测时不一定有的数据时
## to think about target leakage in terms of the timing or chronological order that data becomes available
## 为了预防这种情况发生，任何在目标变量生成后才输入或改变的变量都应被剔除。

#%%  2. Train-Test Contamination 当validation data影响预处理行为
## 用pipeline来预处理数据可解决

#%% 信用卡数据 
import pandas as pd

# Read the data
data = pd.read_csv('C:/Users/Jade Tong/Desktop/KAGGLE/2. Intermediate Machine Learning/AER_credit_card_data.csv', 
                   true_values = ['yes'], false_values = ['no'])

# Select target
y = data.card      ##是否有信用卡

# Select predictors
X = data.drop(['card'], axis=1)

print("Number of rows in the dataset:", X.shape[0])
X.head()

#%% 是个小数据，要用交叉验证来确保准确性
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Since there is no preprocessing, we don't need a pipeline (used anyway as best practice!)
my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))
cv_scores = cross_val_score(my_pipeline, X, y, 
                            cv=5,
                            scoring='accuracy')

print("Cross-validation accuracy: %f" % cv_scores.mean())
#Cross-validation accuracy: 0.981052

#%% 上面说模型准确率为98.11%，很少见有如此高的准确率，应该看一下target leakage
#  看一下变量，发现expenditure不知道表示是在这张卡上的消费还是之前的消费

expenditures_cardholders = X.expenditure[y]   #y=true,即有信用卡的支出
expenditures_noncardholders = X.expenditure[~y]   #y=false,即没有办信用卡的支出

#  没有信用卡同时支出为零的比率
print('Fraction of those who did not receive a card and had no expenditures: %.2f' \
      %((expenditures_noncardholders == 0).mean()))   #1.00
#  有信用卡同时支出为零的比率
print('Fraction of those who received a card and had no expenditures: %.2f' \
      %(( expenditures_cardholders == 0).mean()))     #0.02

#所有没持有信用卡的都没有支出，只有2%持有信用卡的没有支出。
#this also seems to be a case of target leakage, where expenditures probably means expenditures on the card they applied for.

#%% 
# Drop leaky predictors from dataset
potential_leaks = ['expenditure', 'share', 'active', 'majorcards']
X2 = X.drop(potential_leaks, axis=1)

# Evaluate the model with leaky predictors removed
cv_scores = cross_val_score(my_pipeline, X2, y, 
                            cv=5,
                            scoring='accuracy')

print("Cross-val accuracy: %f" % cv_scores.mean())
# Cross-val accuracy: 0.838504

