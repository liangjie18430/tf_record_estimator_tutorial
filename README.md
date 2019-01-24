# 获取对应的jar包到本地文件
* git clone https://github.com/tensorflow/ecosystem.git
* cd ../../hadoop
* mvn versions:set -DnewVersion=1.12.0
* mvn clean install
* cd ../spark/spark-tensorflow-connector
* mvn versions:set -DnewVersion=1.12.0
* mvn clean install -Dspark.version=2.1.1

# 使用run.sh执行对应的py文件
* 产生tfrecord文件
sh +x run.sh data_to_tfrecord.py
直接产出pmml文件:
sh +x run.sh train_pmml_only.py

# how to debug in local
In order to debug convenient, could add all jar to $SPARK_HOME/jars


# 多个tfrecord示例
https://www.programcreek.com/python/example/90440/tensorflow.Example

# references
 并行读取:
 https://blog.csdn.net/u014061630/article/details/80776975
 tf.contrib.data.parallel_interleave



