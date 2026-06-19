# -*- coding: utf-8 -*-
"""Config
This script holds a series of configuration values necessary for executing this repository.
"""
import os 
import datetime as dt
from dotenv import load_dotenv
load_dotenv()

current_date = str(dt.datetime.now())[0:10]
ENV = os.environ.get('ENV', 'STAGING')
creds = {
    'db_creds' : {
        'user' : os.environ.get('DB_USERNAME'),
        'pw' : os.environ.get('DB_PASSWORD'),
        'host' : os.environ.get('DB_HOST')
    },
    's3_creds' : {
        'access_key' : os.environ.get('AWS_ACCESS_KEY_ID'),
        'secret_key' : os.environ.get('AWS_SECRET_ACCESS_KEY'),
        'region' : os.environ.get('AWS_REGION'),
    }
}
