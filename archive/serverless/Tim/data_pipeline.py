import json
import boto3
import os


def handler_aisles(event, context):
    glue_client = boto3.client('glue')
    s3_event = event['Records'][0]['s3']
    
    bucket_name = s3_event['bucket']['name']
    object_key = s3_event['object']['key']

    glue_job_name = 'job_aisles_to_parquet'
    
    try:
        response = glue_client.start_job_run(
            JobName= glue_job_name,
            Arguments={
                '--S3_BUCKET': bucket_name,
                '--S3_KEY': object_key
            }
        )
        return {
            'statusCode': 200,
            'body': json.dumps('Glue job started successfully!')
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps('Error starting Glue job: ' + str(e))
        }
    
def handler_departments(event, context):
    glue_client = boto3.client('glue')
    s3_event = event['Records'][0]['s3']
    
    bucket_name = s3_event['bucket']['name']
    object_key = s3_event['object']['key']

    glue_job_name = 'job_departments_to_parquet'
    
    try:
        response = glue_client.start_job_run(
            JobName= glue_job_name,
            Arguments={
                '--S3_BUCKET': bucket_name,
                '--S3_KEY': object_key
            }
        )
        return {
            'statusCode': 200,
            'body': json.dumps('Glue job started successfully!')
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps('Error starting Glue job: ' + str(e))
        }

def handler_products(event, context):
    glue_client = boto3.client('glue')
    s3_event = event['Records'][0]['s3']
    
    bucket_name = s3_event['bucket']['name']
    object_key = s3_event['object']['key']

    glue_job_name = 'job_products_to_parquet'
    
    try:
        response = glue_client.start_job_run(
            JobName= glue_job_name,
            Arguments={
                '--S3_BUCKET': bucket_name,
                '--S3_KEY': object_key
            }
        )
        return {
            'statusCode': 200,
            'body': json.dumps('Glue job started successfully!')
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps('Error starting Glue job: ' + str(e))
        }
    
def handler_orders(event, context):
    glue_client = boto3.client('glue')
    s3_event = event['Records'][0]['s3']
    
    bucket_name = s3_event['bucket']['name']
    object_key = s3_event['object']['key']

    glue_job_name = 'job_orders_to_parquet'
    
    try:
        response = glue_client.start_job_run(
            JobName= glue_job_name,
            Arguments={
                '--S3_BUCKET': bucket_name,
                '--S3_KEY': object_key
            }
        )
        return {
            'statusCode': 200,
            'body': json.dumps('Glue job started successfully!')
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps('Error starting Glue job: ' + str(e))
        }

def handler_orderproducts(event, context):
    glue_client = boto3.client('glue')
    s3_event = event['Records'][0]['s3']
    
    bucket_name = s3_event['bucket']['name']
    object_key = s3_event['object']['key']

    glue_job_name = 'job_orderproducts_to_parquet'
    
    try:
        response = glue_client.start_job_run(
            JobName= glue_job_name,
            Arguments={
                '--S3_BUCKET': bucket_name,
                '--S3_KEY': object_key
            }
        )
        return {
            'statusCode': 200,
            'body': json.dumps('Glue job started successfully!')
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps('Error starting Glue job: ' + str(e))
        }