

#https://blog.csdn.net/wang263334857/article/details/81836578

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as seb

#读取数据
data  = pd.read_csv("day.csv")
#查看前五行数据
#print(data.head())
#print(data.info())
#print(data.describe())

cate_features = ["season","weathersit","weekday"]
for col in cate_features:
    print("%s属性的不同取值和次数:" % col)
    print(data[col].value_counts())
    data[col] = data[col].astype('object')


#该4类特征的取值不多，用one-hot编码

#特征处理
x_train_cat = data[cate_features]
x_train_cat = pd.get_dummies(x_train_cat)
x_train_cat.head()
df = pd.DataFrame(x_train_cat)
df.to_csv("meng.csv")
print(x_train_cat.head())

#对数值型变量进行处理

#对数据进行归一化处理
# from  sklearn.preprocessing import MinMaxScaler
# mn_x  = MinMaxScaler()
# numerical_features = ["temp","hum","windspeed"]
# temp = mn_x.fit_transform(data[numerical_features])
# x_train_num = pd.DataFrame(data=temp,columns=numerical_features)
# print(x_train_num.head())
########################
from sklearn.preprocessing import PolynomialFeatures #用多项式做数值型数据处理
numerical_features = ["temp","hum","windspeed"]
poly = PolynomialFeatures(degree=4, include_bias=False, interaction_only=False)
X_ploly = poly.fit_transform(data[numerical_features])
X_ploly_df = pd.DataFrame(X_ploly, columns=poly.get_feature_names())
print(X_ploly_df)


########################

#将前边的特征值和4种数值值进行拼接生成一个新的data结果集
y_cols = 'casual'
x_train = pd.concat([x_train_cat,X_ploly_df,data['holiday'],data['workingday']],axis=1,ignore_index=False)
df = pd.DataFrame(x_train)
df.to_csv("meng2.csv")
final_train = pd.concat([data['instant'],x_train,data['yr'],data[y_cols]],axis=1,ignore_index=False)

df = pd.DataFrame(final_train)
df.to_csv("final.csv",index=False)
final_train.head()

#加载生成的特征csv

tz_data = pd.read_csv("final.csv")

train=tz_data[tz_data.yr==0] #训练数据

train = train.drop(columns = ['instant','yr'])
print("train（训练）:"+str(train.shape))



#取2012年的数据作为测试数据
test=tz_data[tz_data.yr==1] #测试数据
#取testID备份留作后用
testID=test['instant']
# testCNT=test[y_cols]

test = test.drop(columns = ['instant','yr'])
print("test（测试）:"+str(test.shape))
print(test.head())



#准备训练数据
# y_cols = 'cnt'
y_cols = 'casual'
#训练数据
y_train = train[y_cols]
X_train = train
X_train = X_train.drop(columns=[y_cols])
#测试数据
y_test_real = test[y_cols]
y_test = test[y_cols]
X_test = test
X_test = X_test.drop(columns = [y_cols])

print(X_train.shape)
print(X_test.shape)

#数据标准化

from sklearn import preprocessing
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

# lgb
params = {'num_leaves': 38,
          'min_data_in_leaf': 50,
          'objective': 'regression',          # 回归设置
          'max_depth': -1,
          'learning_rate': 0.02,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,
          "bagging_freq": 1,
          "bagging_fraction": 0.7,
          "bagging_seed": 11,
          "lambda_l1": 0.1,
          "verbosity": -1,
          "nthread": 4,
          'metric': 'mae',         # 回归设置
          "random_state": 2019,
          # 'device': 'gpu'
          }

### FIT LGBM WITH POISSON LOSS ### 

trn_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

model = lgb.train(params, trn_data, num_boost_round=1000,
                  valid_sets = [trn_data, test_data],
                  verbose_eval=50, early_stopping_rounds=150)

train_y_pred = model.predict(X_train)
test_y_pred =model.predict(X_test)

X_train.shape
X_test.shape




