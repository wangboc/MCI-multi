
注：一般情况，可直接运行Classification.m，而不用DataPrepare。因为BCT数据已经计算好。

如需重新计算BCT，或选择其他特征，则需按照此说明，由1至2顺序执行。


1、首先调用DataPrepare.m 脚本，计算BCT各种特征值，生成数据文件。

2、调用Classification.m 脚本，实现Filter and Wrapper 特征选择过程，并训练模型。

目前实现的机器学习方法有两种：

一是

借助Matlab Machine Learning工具箱，对Selected_train_data进行处理。该变量在计算结束后会存储在
工作区，数据格式为：最后一列是标签列，前面所有列为特征列。

二是

调用libSVM，该工具会自动选择合适的C、G参数，并输出结果。



