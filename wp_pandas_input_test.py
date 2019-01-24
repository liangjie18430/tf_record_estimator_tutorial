# -* coding:utf8 *-
from __future__ import print_function
import pandas as pd
from tensorflow.python.estimator.run_config import RunConfig
import shutil
import datetime
from sklearn import datasets
from model_utils import build_estimator
from sklearn.model_selection import train_test_split
import numpy as np
import time
import tensorflow as tf

tf.set_random_seed(12)
run_config = RunConfig(tf_random_seed=12)


def get_breast_cancer_df():
    breast_cancer = datasets.load_breast_cancer()
    x = breast_cancer.data
    y = breast_cancer.target
    feature_names = [name.replace(' ', '') for name in breast_cancer.feature_names]
    x_pd = pd.DataFrame(x, columns=feature_names)
    y_pd = pd.DataFrame(y, columns=['label'])
    all_pd = pd.concat([x_pd, y_pd], axis=1)
    return all_pd


# 根据不同的列构建不同的tf_column的输入
def construct_tf_column(continuous_result_columns):
    # 如果是连续的列，肯定是实质性
    con_tf_columns = []
    for column in continuous_result_columns:
        tf_column = tf.feature_column.numeric_column(key=column)
        con_tf_columns.append(tf_column)

    deep_columns = con_tf_columns
    # 对于wide部分，直接传入交叉特征,和其他的离散特征
    wide_columns = con_tf_columns
    return wide_columns, deep_columns


def input_fn(df, batch_size=500, num_epochs=2, shuffle=True, num_threads=5):
    data_df = df.drop(["label"], axis=1)
    label_df = df["label"]
    return tf.estimator.inputs.pandas_input_fn(x=data_df, y=label_df, num_epochs=num_epochs, shuffle=shuffle,
                                               batch_size=batch_size, num_threads=num_threads)


def get_base_columns(df):
    if hasattr(df, "columns"):
        df_columns = df.columns
    else:
        raise ValueError("df has no attr columns")
    continuous_result_columns = []
    discrete_result_columns = []
    for column in df_columns:
        continuous_result_columns.append(column)

    continuous_result_columns.remove("label")
    # 查看真正的哪些column没有被匹配上
    all_result_columns = set(continuous_result_columns).union(set(discrete_result_columns))
    other_columns = set(df_columns).difference(all_result_columns)
    print("the columns that not match:\n", other_columns)
    return continuous_result_columns, discrete_result_columns


def model_metric(model, df, message):
    """
    model为训练好的模型
    df为需要评估的dataframe
    message 为需要输出的额外信息

    """
    start_time = time.clock()
    results = model.evaluate(input_fn=input_fn(df, num_epochs=1, shuffle=False))
    end_time = time.clock()
    print(message, "eval time cost: %f s" % (end_time - start_time))
    # Display evaluation metrics,输出每次的评估结果
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))
    pass


def main():
    train_start_time = datetime.datetime.now()
    all_pd = get_breast_cancer_df()
    train_df, test_df = train_test_split(all_pd, stratify=all_pd['label'], random_state=42, test_size=0.2)
    # 获取离散的列和非离散的列
    continuous_result_columns, discrete_result_columns = get_base_columns(train_df)

    model_dir = "/Users/admin/workspace/test_workspace/tboard/wp"
    shutil.rmtree(model_dir, ignore_errors=True)
    wide_columns, deep_columns = construct_tf_column(continuous_result_columns)

    # model = build_estimator(wide_columns=wide_columns,deep_columns=deep_columns,model_type="deep")
    model = build_estimator(deep_columns = wide_columns,wide_columns=wide_columns,  model_type='wide_deep')
    # model = build_estimator(wide_columns=wide_columns, deep_columns=deep_columns, model_dir=model_dir)
    model.train(input_fn=input_fn(train_df, num_epochs=3, shuffle=False))
    model_metric(model, train_df, "train metric.")
    model_metric(model, test_df, "test metric.")
    train_end_time = datetime.datetime.now()
    print("all done time", str(train_end_time - train_start_time))


def main_test_input():
    train_start_time = datetime.datetime.now()
    all_pd = get_breast_cancer_df()
    train_df, test_df = train_test_split(all_pd, stratify=all_pd['label'], random_state=42, test_size=0.2)
    print("train_df.shape: ", train_df.shape, ", test_df.shape: ", test_df.shape)
    input_test = input_fn(train_df, num_epochs=1, shuffle=False, batch_size=100,num_threads=1)
    features_batch, label_batch = input_test()
    count = 0
    with tf.Session() as sess:
        # 开启一个协调器
        coord = tf.train.Coordinator()
        # 使用start_queue_runners 启动队列填充
        threads = tf.train.start_queue_runners(sess, coord)
        try:
            while not coord.should_stop():
                print('************')
                # 获取每一个batch中batch_size个样本和标签
                features, label = sess.run([features_batch, label_batch])
                print("features_shape: ", str(np.shape(features)), ", label_shape: " + str(np.shape(label)))
                count = count + len(label)
                # print("features: ", str(features), ", label: ", str(label))

        except tf.errors.OutOfRangeError:  # 如果读取到文件队列末尾会抛出此异常
            print("read to the last done! now lets kill all the threads……")
        finally:
            # 协调器coord发出所有线程终止信号
            coord.request_stop()
            print('all threads are asked to stop!')
        coord.join(threads)  # 把开启的线程加入主线程，等待threads结束
    print("all count is: " + str(count))
    train_end_time = datetime.datetime.now()
    print("all done time", str(train_end_time - train_start_time))


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    main()
    # main_test_input()

