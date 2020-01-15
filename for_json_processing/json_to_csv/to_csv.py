"""
Created on  1/9/20
@author: Jingchao Yang

http://blog.appliedinformaticsinc.com/how-to-parse-and-convert-json-to-csv-using-python/
"""
import json
import os
import csv
from datetime import datetime

env_path = '../'


def json_readr(path, file):
    """

    :param path:
    :param file:
    :return:
    """
    fname = os.path.join(path, file)
    # open a file for writing
    out_file = open(os.path.join(path, 'processed/' + file + '.csv'), 'w')
    # create the csv writer object
    csvwriter = csv.writer(out_file)
    print('start writing:', file + '.csv')
    count = 0

    for line in open(fname, mode="r"):
        data = json.loads(line)
        # coor = (data['latitude'], data['longitude'])
        hourlt_data = data['hourly']['data']
        for rec in hourlt_data:
            rec['time'] = (datetime.utcfromtimestamp(rec['time']).strftime('%Y-%m-%d %H:%M:%S'))
            if count == 0:
                header = rec.keys()
                csvwriter.writerow(header)
                count += 1

            csvwriter.writerow(rec.values())
    out_file.close()


all_coor = []
for f in os.listdir(env_path):
    if f.endswith(".txt"):
        json_readr(env_path, f)
