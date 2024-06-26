import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job

from pyspark.sql.types import StructType, StructField, BooleanType, ByteType, ShortType, IntegerType, StringType, FloatType, DoubleType
  
sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)


# Database and table names
#source_database = "cfn_imba_database_csv"
source_table = "orders"
target_path = "s3://tgou1055-imba-sls-event/data_parquet/orders/"

# Define table schema
orders_schema = StructType([
    StructField("order_id", IntegerType(), True),
    StructField("user_id", IntegerType(), True),
    StructField("eval_set", StringType(), True),
    StructField("order_number", IntegerType(), True),
    StructField("order_dow", IntegerType(), True),
    StructField("order_hour_of_day", IntegerType(), True),
    StructField("days_since_prior", DoubleType(), True)
])

# Convert csv format to parquet format
orders_df = spark.read.csv("s3://tgou1055-imba-sls-event/data_csv/orders/orders.csv", \
                                 header=True, schema=orders_schema)
orders_df.write.mode("overwrite").parquet(target_path)

# conclude session
spark.stop()
job.commit()