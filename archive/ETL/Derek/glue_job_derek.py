import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job



def main():

    # create glue context first
    glueContext = GlueContext(SparkContext.getOrCreate())
    
    # TODO ---------------------------------------------------
    # write codes to produce pyspark dataframes for up_features,prd_features,user_features_1, user_features_2 according to the sql queries you written.
    # write the dataframe to s3 location with parquet format (e.g. write up_feature dataframe to s3://<your s3 bucket>/features/up_feature/)
    from pyspark.sql.functions import col, min, max, sum, avg, count, countDistinct, row_number
    from pyspark.sql.window import Window
    from pyspark.sql.types import StructType, StructField, BooleanType, ByteType, ShortType, IntegerType, StringType, FloatType, DoubleType

    sc = SparkContext.getOrCreate()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    job = Job(glueContext)
    
    #aisles
    aisles_schema = StructType([
    StructField("aisle_id", IntegerType(), True),
    StructField("aisle", StringType(), True)
    ])
    aisles = spark.read.csv("s3://imba-derek/data/aisles/", header=True, schema=aisles_schema)
    aisles.write.mode("overwrite").parquet("s3://derek-raw-parquet/aisles/")
    aisles = spark.read.parquet('s3://derek-raw-parquet/aisles/')
    aisles.printSchema()
    print(f'row count: {aisles.count()}')
    
    #departments
    departments_schema = StructType([
    StructField("department_id", IntegerType(), True),
    StructField("department", StringType(), True)
    ])
    departments = spark.read.csv("s3://imba-derek/data/departments/departments.csv", header=True, schema=departments_schema)
    departments.write.mode("overwrite").parquet("s3://derek-raw-parquet/departments/")
    departments = spark.read.parquet('s3://derek-raw-parquet/departments') # read as parquet
    departments.printSchema()
    print(f'row count: {departments.count()}')
    
    #products
    products_schema = StructType([
    StructField("product_id", IntegerType(), True),
    StructField("product_name", StringType(), True),
    StructField("aisle_id", IntegerType(), True),
    StructField("department_id", IntegerType(), True)
    ])
    products = spark.read.csv("s3://imba-derek/data/products/products.csv", header=True, schema=products_schema)
    products.write.mode("overwrite").parquet("s3://derek-raw-parquet/products/")
    products = spark.read.parquet('s3://derek-raw-parquet/products') # read as parquet
    products.printSchema()
    print(f'row count: {products.count()}')
    
    #denorm products
    products_denorm = products\
                    .join(aisles, products.aisle_id==aisles.aisle_id, 'inner')\
                    .join(departments, products.department_id==departments.department_id, 'inner')\
                    .select(products.product_id,
                            products.product_name,
                            products.aisle_id,
                            aisles.aisle,
                            products.department_id,
                            departments.department
                           )
    products_denorm.printSchema()
    products_denorm.write.mode("overwrite").parquet("s3://derek-transformed-data/products/")
        
    #orders
    orders_schema = StructType([
    StructField("order_id", IntegerType(), True),
    StructField("user_id", IntegerType(), True),
    StructField("eval_set", StringType(), True),
    StructField("order_number", IntegerType(), True),
    StructField("order_dow", ByteType(), True),
    StructField("order_hour_of_day", ByteType(), True),
    StructField("days_since_prior_order", FloatType(), True)
    ])
    orders = spark.read.csv("s3://imba-derek/data/orders/orders.csv", header=True, schema=orders_schema)
    orders.write.partitionBy("eval_set").mode("overwrite").parquet("s3://derek-raw-parquet/orders/")
    orders = spark.read.parquet('s3://derek-raw-parquet/orders') # read as parquet
    orders.printSchema()
    print(f'row count: {orders.count()}')
    orders.agg(min('order_number'), max('order_number')).show()
    orders.agg(min('days_since_prior_order'), max('days_since_prior_order')).show()
    
    # filter by eval_set=prior
    orders_prior = orders.where(orders.eval_set=='prior').select(*[c for c in orders.columns if c!='eval_set'])
    print(f'row count: {orders_prior.count()}')
    orders_prior.write.mode("overwrite").parquet("s3://derek-transformed-data/orders_prior/")
    
    # takes 1 minute to run
    order_products_schema = StructType([
        StructField("order_id", IntegerType(), True),
        StructField("product_id", IntegerType(), True),
        StructField("add_to_cart_order", IntegerType(), True),
        StructField("reordered", IntegerType(), True)
    ])
    order_products = spark.read.csv("s3://imba-derek/data/order_products/", header=True, schema=order_products_schema)
    order_products = order_products.withColumn("reordered", col("reordered").cast("boolean"))
    order_products.write.mode("overwrite").parquet("s3://derek-raw-parquet/order_products/")
    order_products = spark.read.parquet('s3://derek-raw-parquet/order_products') # read as parquet
    order_products.printSchema()
    print(f'row count: {order_products.count()}')
    
    # takes 20 seconds to run
    order_products_prior = orders_prior\
                            .join(order_products, orders_prior.order_id==order_products.order_id, 'inner')\
                            .select(orders_prior.order_id,
                                    orders_prior.user_id,
                                    orders_prior.order_number,
                                    orders_prior.order_dow,
                                    orders_prior.order_hour_of_day,
                                    orders_prior.days_since_prior_order,
                                    order_products.product_id,
                                    order_products.add_to_cart_order,
                                    order_products.reordered
                                )
    order_products_prior.write.mode("overwrite").parquet("s3://derek-transformed-data/order_products_prior/")
        
    # orders = spark.read.parquet('s3://derek-raw-parquet/orders/') # read as parquet
    user_features_1 = orders.groupBy('user_id').agg(max('order_number').alias('max_order_number'),
                                               sum('days_since_prior_order').alias('sum_days_since_prior_order'),
                                               avg('days_since_prior_order').alias('avg_days_since_prior_order')
                                               )
    user_features_1.orderBy('user_id').show(5)
    print(f'row count: {user_features_1.count()}')
    # save aggregated result as one part
    user_features_1.write.mode("overwrite").parquet("s3://derek-transformed-data/user_features_1/")
        
    # order_products_prior = spark.read.parquet('s3://derek-transformed-data/order_products_prior/') # read as parquet
    
    user_features_2 = order_products_prior.groupBy('user_id').agg(count('product_id').alias('total_products'),
                                                              countDistinct('product_id').alias('total_distinct_products'),
                                                              (sum(col('reordered').cast('int'))/
                                                               sum((col('order_number')>1).cast('int'))).alias('reorder_ratio')
                                                            )
    user_features_2.orderBy('user_id').show(5)
    print(f'row count: {user_features_2.count()}')
    user_features_2.write.mode("overwrite").parquet("s3://derek-transformed-data/user_features_2/")
        
    up_features = order_products_prior.groupBy('user_id', 'product_id').agg(count('order_id').alias('total_orders'),
                                                                        min('order_number').alias('min_order_number'),
                                                                        max('order_number').alias('max_order_number'),
                                                                        avg('add_to_cart_order').alias('avg_add_to_cart_order')
                                                                       )
    up_features.orderBy('user_id', 'product_id').show(5)
    print(f'row count: {up_features.count()}')
    up_features.write.mode("overwrite").parquet("s3://derek-transformed-data/up_features/")
            
    prod_seq = order_products_prior.withColumn('product_seq_time', 
                                           row_number().over(Window\
                                                             .partitionBy('user_id', 'product_id')\
                                                             .orderBy(col('order_number').asc())
                                                            )
                                          ).select('product_id', 'reordered', 'product_seq_time')

    prd_features = prod_seq.groupBy('product_id').agg(count('product_id').alias('total_products'),
                                                    sum(col('reordered').cast('int')).alias('total_reordered'),
                                                    sum((col('product_seq_time')==1).cast('int')).alias('product_seq_time_is_1'),
                                                    sum((col('product_seq_time')==2).cast('int')).alias('product_seq_time_is_2')
                                                    )
    prd_features.orderBy('product_id').show(5)
    print(f'row count: {prd_features.count()}')
    prd_features.write.mode("overwrite").parquet("s3://derek-transformed-data/prd_features/")        
            
    # END TODO ---------------------------------------------------
    
    # creating dataframes from existing athena catelog
    up_features = glueContext.create_dynamic_frame_from_options(connection_type = "parquet", connection_options = {"paths": ["s3://derek-project-gluejob/features/up_feature/"]})
    prd_features = glueContext.create_dynamic_frame_from_options(connection_type = "parquet", connection_options = {"paths": ["s3://derek-project-gluejob/features/prd_feature/"]})
    user_features_1 = glueContext.create_dynamic_frame_from_options(connection_type = "parquet", connection_options = {"paths": ["s3://derek-project-gluejob/features/user_features_1/"]})
    user_features_2 = glueContext.create_dynamic_frame_from_options(connection_type = "parquet", connection_options = {"paths": ["s3://derek-project-gluejob/features/user_features_2/"]})
    
    # join user features together
    users = Join.apply(user_features_1.rename_field('user_id','user_id1'), user_features_2, 'user_id1', 'user_id').drop_fields(['user_id1'])
    
    # join everything together
    df = Join.apply(Join.apply(up_features, 
                      users.rename_field('user_id','user_id1'), 
                      'user_id','user_id1').drop_fields(['user_id1']),
          prd_features.rename_field('product_id','product_id1'), 
          'product_id','product_id1').drop_fields(['product_id1'])
          
    # convert glue dynamic dataframe to spark dataframe
    df_spark = df.toDF()
    df_spark.repartition(1).write.mode('overwrite').format('csv').save("s3://derek-transformed-data/output", header = 'true')
    
if __name__ == '__main__':
    main()