
import tensorflow as tf
#tf.__version__
#tf.test.is_gpu_available()
import sys
sys.path.append('test/KwaiSurvival')
import pandas as pd
from DeepSurv import *
from DeepHit import *
from DeepMultiTasks import *

df = pd.read_csv('test/KwaiSurvival/demo/example_data.csv')
label = 'Time'
event = 'Event'

# 初始化 - 第一个模型 DeepSurv
ds = DeepSurv(df,label, event)

# 初始化 - 第二个模型 DeepHit
ds = DeepHit(df,label, event)

# 初始化 - 第三个模型 DeepMultiTasks
dm = DeepMultiTasks(df,label, event)

# 训练
epochs = 100      
ds.train( epochs)

epochs = 10   
dm.train( epochs)


# 模型summary
ds.model.summary()
ds.X.shape

# 模型保存 - 这里代码没写，感觉这版代码是个实验版本的
ds.model.save('test/KwaiSurvival/path_to_my_model_DeepSurv.h5')
ds.model.save('test/KwaiSurvival/path_to_my_model_DeepHit.h5')
ds.model.save('test/KwaiSurvival/path_to_my_model_DeepMultiTasks.h5')


# 预测
scores = ds.predict_score(ds.X) # 1019 ,4
df['pred'] = scores

# concordance_eval 评估
ds.concordance_eval( X = None, event_times = None, event_observed = None)
ds.concordance_eval( X = ds.X, event_times = ds.label, event_observed = ds.event)


# ds.X.shape  # (1019, 4)
# ds.label.shape # (1019,)
# ds.event.shape # (1019,)
# df[label][:4].shape
    #  局部一致性，注意格式不然报错
ds.concordance_eval( X = df.iloc[:4,:4].to_numpy(), event_times = df[label][:4], event_observed =  df[event][:4])

# 完成DeepSurv模型对于X数据集的survival function 预测
survival_function = ds.predict_survival_func( ds.X)
survival_function.shape # 100*1019

'''
会输出一个矩阵：100*1019

从`predict_survival_func`函数中这句得知：
data = ds.df.groupby(label, as_index=False).agg({event: 'sum', 'partial_hazard': "sum"}).sort_values(label,
																										 ascending=False)

这里的100代表所有的时间点，也就是：len(set(ds.label))
1019代表所有X个样本

也就是1019个样本，在所有时间点的留存率,是呈现递减的方式的

'''

# 第1000个数据的，KM曲线 - 上面的可视化
ds.plot_survival_func(ds.X, 1000)










