# -* coding:utf8 *-
from __future__ import print_function
import pandas as pd

from pyspark.sql import SparkSession, DataFrame
m __future__ import print_function
import pandas as pd

from pyspark.sql import SparkSession
from pyspark import SparkConf
import os
from sklearn.model_selection import train_test_split


def create_spark():
    spark_conf = SparkConf().setAppName("Spark_TF_Offline_File") \
        .set("spark.hadoop.validateOutputSpecs", "false") \
        .set("spark.driver.maxResultSize", "20g") \
        .set("spark.executor.heartbeatInterval", "10000") \
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .setMaster("local")

    spark_session = SparkSession.builder.config(conf=spark_conf) \
        .getOrCreate()
    return spark_session


def get_breast_cancer_df():
    from sklearn import datasets
    breast_cancer = datasets.load_breast_cancer()
    x = breast_cancer.data
    y = breast_cancer.target
    feature_names = [name.replace(' ', '_') for name in breast_cancer.feature_names]
    x_pd = pd.DataFrame(x, columns=feature_names)
    y_pd = pd.DataFrame(y, columns=['label'])
    all_pd = pd.concat([x_pd, y_pd], axis=1)
    return all_pd


def save_df_to_tf(df, file_path):
    df.write.format("tfrecords").option("recordType", "Example").save(file_path)
    print("save to file_path: ", file_path)


def trans_pd_df_to_sp_df(spark, df):
    spark_df = spark.createDataFrame(df)
    return spark_df


def main():
    base_path = "/Users/admin/workspace/data/train_data"
    spark = create_spark()
    all_pd = get_breast_cancer_df()

    train_df, test_df = train_test_split(all_pd, stratify=all_pd['label'], random_state=42, test_size=0.2)

    spark_train_df = trans_pd_df_to_sp_df(spark, train_df)
    spark_test_df = trans_pd_df_to_sp_df(spark, test_df)
    # save df to df
    #
    train_save_file_path = base_path + os.sep + "breast_tf_save_train"
    test_save_file_path = base_path + os.sep + "breast_tf_save_test"
    save_df_to_tf(spark_train_df, train_save_file_path)
    save_df_to_tf(spark_test_df, test_save_file_path)

    print("save train path: ", train_save_file_path)
    print("save test path: ", test_save_file_path)


if __name__ == '__main__':
    main()

from pyspark import SparkConf, SparkContext
import os

def create_spark():
    spark_conf = SparkConf().setAppName("Spark_TF_Offline_File") \
        .set("spark.hadoop.validateOutputSpecs", "false") \
        .set("spark.driver.maxResultSize", "20g") \
        .set("spark.executor.heartbeatInterval", "10000") \
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .setMaster("local")

    spark_session = SparkSession.builder.config(conf=spark_conf) \
        .getOrCreate()
    return spark_session

def get_breast_cancer_df():

    from sklearn import datasets
    breast_cancer = datasets.load_breast_cancer()
    x = breast_cancer.data
    y = breast_cancer.target
    feature_names = [name.replace(' ', '_') for name in breast_cancer.feature_names]
    x_pd = pd.DataFrame(x, columns=feature_names)
    y_pd = pd.DataFrame(y, columns=['label'])
    all_pd = pd.concat([x_pd, y_pd], axis=1)
    return all_pd


def save_df_to_tf(df, file_path):
    df.write.format("tfrecords").option("recordType", "Example").save(file_path)
    print("save to file_path: ", file_path)





def trans_pd_df_to_sp_df(spark,df):
    spark_df = spark.createDataFrame(df)
    return spark_df


def main():
    spark = create_spark()
    df = get_breast_cancer_df()
    spark_df = trans_pd_df_to_sp_df(spark,df)

    # save df to df
    #
    save_file_path = "/Users/admin/workspace/data/train_data" + os.sep+"breast_tf_save"
    save_df_to_tf(spark_df,save_file_path)

    print("save file path: ",save_file_path)
if __name__ == '__main__':

    main()




