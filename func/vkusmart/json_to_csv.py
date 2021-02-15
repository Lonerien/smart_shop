import os
import json
import jsonpickle
from collections import defaultdict
from vkusmart_release.history import HistoryEntry 
from typing import DefaultDict
import csv

''' запись в .csv:
track_id: был_на_кассе, взято товаров, время входа, время выхода, время_начала_оплаты, время-конца оплаты
'''

def json_to_class(info_dir):
    history: DefaultDict[int, HistoryEntry] = defaultdict(HistoryEntry)
    for persons_id in os.listdir(info_dir):
        json_filename = info_dir + '/' + persons_id + 'info.json'
        with open(json_filename, 'r') as infile:
            history_instance_from_json = json.load(infile)
        history[int(persons_id)] = jsonpickle.decode(history_instance_from_json)     
    return history

def class_to_csv(history):
    csvData = [['track_id', 'is_paid', 'pick_count', 'time_in', 'time_out', 'time_pay_begin', 'time_pay_end']]
    for track_id, track_info in history.items():
        csvData.append( 
                   track_id, 
                   track_info.is_cassa_visited,
                   track_info.pick_count,
                   track_info.store_enter,
                   track_info.store_exit,
                   track_info.cassa_enter,
                   track_info.cassa_exit
        )
                   
    with open(info_dir +'persons_important.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    
    
    
