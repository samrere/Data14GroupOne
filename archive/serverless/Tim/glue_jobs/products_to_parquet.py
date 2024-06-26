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
source_table = "products"
target_path = "s3://tgou1055-imba-sls-event/data_parquet/products/"

# Define table schema
products_schema = StructType([
    StructField("product_id", IntegerType(), True),
    StructField("product_name", StringType(), True),
    StructField("aisle_id", IntegerType(), True),
    StructField("department_id", IntegerType(), True)
])

# Convert csv format to parquet format
products_df = spark.read.csv("s3://tgou1055-imba-sls-event/data_csv/products/products.csv", \
                                 header=True, schema=products_schema)
products_df.write.mode("overwrite").parquet(target_path)

# conclude session
spark.stop()
job.commit()