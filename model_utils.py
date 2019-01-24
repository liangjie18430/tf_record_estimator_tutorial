# -* coding:utf8 *-
from __future__ import print_function
import tensorflow as tf


def build_estimator(model_dir=None, model_type="wide_deep", wide_columns=None, deep_columns=None):
    """
    定义构建模型的方法
    :param model_dir:
    :param model_type:
    :return:
    """
    hidden_units = [100, 50]

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
    run_config = tf.estimator.RunConfig().replace(session_config=tf_config, tf_random_seed=12)
    if model_type == 'wide':
        # 使用线性模型
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns,
            config=run_config)
    elif model_type == 'deep':
        # 使用深度学习模型
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            config=run_config)
    else:
        # 使用wide and deep模型
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units,
            config=run_config)

