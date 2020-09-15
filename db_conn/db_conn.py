"""
Created on  9/14/2020
@author: Jingchao Yang
"""
import pandas as pd
import psycopg2
from sqlalchemy import types

from sqlalchemy import create_engine

engine = create_engine('postgresql://postgres:cisc255b@35.168.2.254/jy_db')

# df = pd.read_csv(r'D:\harveyTwitter\db_back\user_coor.csv')
# df.to_sql('user_coor',
#           con=engine,
#           if_exists='replace',
#           index=False,
#           dtype={
#               "eid": types.INTEGER,
#               "tlat": types.Float(),
#               "tlng": types.Float(),
#               "tid": types.BIGINT
#           })

table_df = pd.read_sql_table(
    'user_coor',
    con=engine
)
print(table_df)


# dbConnect = "dbname='jy_db' user='postgres' host='35.168.2.254' password='cisc255b'"
# conn = psycopg2.connect(dbConnect)
# cur = conn.cursor()
# cur.execute("""
#     CREATE TABLE users(
#     id integer PRIMARY KEY,
#     email text,
#     name text,
#     address text
# )
# """)
# conn.commit()


# df = pd.read_csv(r'D:\harveyTwitter\db_back\user_coor.csv')
# df.to_sql('user_coor', con=dbConnect, if_exists='append')
