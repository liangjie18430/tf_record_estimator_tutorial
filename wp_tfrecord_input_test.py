# -* coding:utf8 *-
from __future__ import print_function
import os
import tensorflow as tf
import six
from model_utils import build_estimator
import time
import datetime
import numpy as np
import shutil

# https://blog.csdn.net/jacke121/article/details/78823974
# https://blog.csdn.net/jacke121/article/details/78823974
# https://blog.csdn.net/happyhorizion/article/details/77894055
# https://stackoverflow.com/questions/37151895/tensorflow-read-all-examples-from-a-tfrecords-at-once

#
base_path = "/Users/admin/workspace/data/train_data"
# base_path="hdfs://nn-cluster/user/newhouse/warehouse/dm_newhouse/tmp"

file_sep = "\t"

# save_df_dir = base_path + os.sep + "tf_train_with_head"
# save_df_dir = base_path + os.sep + "tf_en_test_with_head"
train_save_df_dir = base_path + os.sep + "breast_tf_save_train"
test_save_df_dir = base_path + os.sep + "breast_tf_save_test"
head_single = train_save_df_dir + os.sep + "part-r-00000"
train_save_df_dir_list = [train_save_df_dir + os.sep + "part-r-00000"]
test_save_df_dir_list = [test_save_df_dir + os.sep + "part-r-00000"]

save_file_path = base_path + os.sep + "breast_tf_save"


def read_features(file_name):
    """
    解析单个的example文件，用于获取愿数据信息
    :return:
    """
    example = tf.train.Example()

    record_iterator = tf.python_io.tf_record_iterator(path=file_name)

    for record in record_iterator:
        if isinstance(example, tf.train.Example):
            example.ParseFromString(record)
            f = example.features

            return f


def get_feature_schema(dictionary):
    """
     生成用于解析example中的features的类型信息，返回值可以用于tf.parse_single_example(example_proto, features)
    :param dictionary: k,v形式的dict，可以从dict中获取对应的读取数据features
    :return:
    """
    features = dict()
    data_type_result = dict()
    for (k, v) in six.iteritems(dictionary):
        if isinstance(v, six.integer_types):
            features[k] = tf.FixedLenFeature([], tf.int64)

            # 此处为tf.int32,上头为tf.int64，是因为example的格式中定义的为int64
            data_type_result[k] = tf.int32
        elif isinstance(v, float):
            features[k] = tf.FixedLenFeature([], tf.float32)
            data_type_result[k] = tf.float32
        elif isinstance(v, six.string_types):
            if not six.PY2:  # Convert in python 3.
                v = [bytes(x, "utf-8") for x in v]
            features[k] = tf.FixedLenFeature([], tf.string)
            data_type_result[k] = tf.string
        elif isinstance(v, bytes):
            features[k] = tf.FixedLenFeature([], tf.string)
            data_type_result[k] = tf.string
        else:
            raise ValueError("Value for %s is not a recognized type; v: %s type: %s" %
                             (k, str(v[0]), str(type(v[0]))))
    # return tf.train.Example(features=tf.train.Features(feature=features))
    return features, data_type_result


def model_metric(model, df, message):
    """
    model为训练好的模型
    df为需要评估的dataframe
    message 为需要输出的额外信息

    """
    start_time = time.clock()
    results = model.evaluate(input_fn=lambda: dataset_input_fn(df, num_epochs=1))
    # results = model.evaluate(input_fn=lambda :input_fn_file(drop_three_filename, num_epochs=1, shuffle=False,feature_columns=train_df.columns,default_value=default_value))
    end_time = time.clock()
    print(message, "eval time cost: %f s" % (end_time - start_time))
    # Display evaluation metrics,输出每次的评估结果
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))


def get_file_list(file_name):
    if isinstance(file_name, str):
        return [file_name]
    elif isinstance(file_name, list) or isinstance(file_name, tuple) or isinstance(file_name, set):
        return list(file_name)


def get_column_name():
    """
    使用breast数据集做的校验，保证和产生的tfrecord中的列名保持一致
    :return:
    """
    from sklearn import datasets
    breast_cancer = datasets.load_breast_cancer()
    feature_names = [name.replace(' ', '_') for name in breast_cancer.feature_names]
    return feature_names


def construct_feature_name(column_names):
    """
    使用feature_column 构建列，用于输入到estimator中
    :param column_names:
    :return:
    """
    con_tf_columns = []
    for column in column_names:
        tf_column = tf.feature_column.numeric_column(key=column)
        con_tf_columns.append(tf_column)
    return con_tf_columns


def get_feature_list_value(feature_k):
    """
    返回从example protobuf下解析出的value值
    :param feature_k: example中features下的feature_list解析出的value值，具体参考tf.train.Example中protobuf格式
    :return:
    """
    is_match = False
    if hasattr(feature_k, "byte_list") and is_match is False:
        value_list = feature_k.byte_list.value
        if len(value_list) != 0:
            value = value_list
            is_match = True
    if hasattr(feature_k, "float_list") and is_match is False:
        value_list = feature_k.float_list.value
        if len(value_list) != 0:
            value = value_list
            is_match = True
    if hasattr(feature_k, "int64_list") and is_match is False:
        value_list = feature_k.int64_list.value
        if len(value_list) != 0:
            value = value_list
            is_match = True
    return value


def parse_feature_schema_from_trecord_file(file_name):
    # 获取特征
    features = read_features(file_name)
    # 获取example对象中features的featuremap
    feature = features.feature;
    # 通过example的格式进行解析,并放入字典中
    parse_result = dict()
    for k in feature:  # feature is a map
        feature_k = feature[k]
        value = get_feature_list_value(feature_k)
        # print("value:", value[0])
        parse_result[k] = value[0]
    # 通过获取到的parse，构建用于解析的features,和对应的数据类型，数据类型用于转换
    feature_schema_result, feature_data_type_result = get_feature_schema(parse_result)
    return feature_schema_result, feature_data_type_result


feature_schema, feature_data_type = parse_feature_schema_from_trecord_file(head_single)


def _parse_function(example_proto):
    # construct schema for other tfrecords

    # 将每个特征文件进行解析
    parsed_features = tf.parse_single_example(example_proto, feature_schema)
    # 将label转换为int
    features = dict()
    # k means key
    for k in feature_data_type:
        temp = tf.cast(parsed_features[k], feature_data_type[k])
        features[k] = temp
    #
    if features.get("label") is not None:
        label = features.pop("label")

    # 将其凑成features和label,features为dict类型，label也为dict类型
    # https://blog.csdn.net/u014061630/article/details/83013402
    return features, label


def dataset_input_fn(dataset, batch_size=500, num_epochs=2):
    """
    注意此处的shuffle和平时的shuffle不一致，此处传入的shuffle的参数为buffer_size
    :return:
    """
    dataset = dataset.repeat(num_epochs).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features_batch, label_batch = iterator.get_next()
    return features_batch, label_batch


def main():
    train_start_time = datetime.datetime.now()
    # file_name = get_file_list(save_df_dir)
    train_files = get_file_list(train_save_df_dir_list)
    test_files = get_file_list(test_save_df_dir_list)

    # https://blog.csdn.net/weixin_42499236/article/details/83998139
    train_dataset = tf.data.TFRecordDataset(train_files)
    train_dataset = train_dataset.map(_parse_function)

    test_dataset = tf.data.TFRecordDataset(test_files)
    test_dataset = test_dataset.map(_parse_function)
    #

    # 使用获得的列名
    base_columns = get_column_name()
    # 获取模型
    wide_columns = construct_feature_name(base_columns)
    model_dir = "test"
    shutil.rmtree(model_dir, ignore_errors=True)

    model = build_estimator(model_dir, model_type="wide_deep", wide_columns=wide_columns, deep_columns=wide_columns)
    # 使用迭代器喂入数据给模型,需要epochs=15,auc可达0。94，设置为3时auc只有0.5

    model.train(input_fn=lambda: dataset_input_fn(train_dataset, num_epochs=15))

    #
    # 模型评估

    model_metric(model, train_dataset, "train metric.")
    model_metric(model, test_dataset, "test metric.")

    train_end_time = datetime.datetime.now()

    print("all done time: ", str(train_end_time - train_start_time))


def main_test_input():
    train_start_time = datetime.datetime.now()
    files = get_file_list(train_save_df_dir_list)
    count = 0
    with tf.Session() as sess:
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(_parse_function)
        # 注意设置batch_size 要小于文件的大小，否则会报错
        result_features, result_features_label = dataset_input_fn(dataset, num_epochs=1, batch_size=100)
        # 开启一个协调器
        coord = tf.train.Coordinator()
        # 使用start_queue_runners 启动队列填充
        threads = tf.train.start_queue_runners(sess, coord)
        try:
            while not coord.should_stop():
                print('************')
                # 获取每一个batch中batch_size个样本和标签
                features, label = sess.run([result_features, result_features_label])
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
    print("the count is: ", count)
    train_end_time = datetime.datetime.now()

    print("all done time: ", str(train_end_time - train_start_time))


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    main()
    # main_test_input()

