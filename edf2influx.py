import numpy as np
import math
import argparse
import os
import pandas as pd
import sys

# For dealing with the EDF files
import mne
import datetime
import pytz
import json

from confluent_kafka import Producer
from functions import *

debug = getattr(sys, 'gettrace', None)()

# Define a callback function to handle delivery reports
def delivery_report(err, msg):
    if err is not None:
        print('Message delivery failed: {}'.format(err))
    else:
        print('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))
    
# Crea el productor de Kafka
# Configuration for Kafka Producer
producer_config = {
    'bootstrap.servers': 'epilsera.ual.es:9092',  # Replace with your Kafka broker(s)
    'client.id': 'edf2influx-producer'
}
producer = Producer(producer_config)

# Print the program title
print('====================')
print('\033[96m \033[1m EDF 2 InfluxDB \033[0m')
print('====================')


# Path to the dataset
dataset_path = 'Y:\datasets\chb-mit'

subject_info = pd.read_csv(dataset_path+"\SUBJECT-INFO", delimiter='\t')

# Buscar la edad y el género del paciente "chb10"
row = subject_info[subject_info["Case"] == 'chb10']
print(row["Gender"].values[0])

print('\nReading EDF files...')
start_time = datetime.datetime.now()

for patient in os.listdir(dataset_path):
    patient_path = os.path.join(dataset_path, patient)
    if os.path.isdir(patient_path):
        sessions = extract_summary(os.path.join(patient_path, patient)+"-summary.txt")
        for file_name in os.listdir(patient_path):
            if file_name.endswith('.edf'):
                if debug:
                    print('\nReading EDF files...')
                    start_time = datetime.datetime.now()
                file_path = os.path.join(patient_path, file_name)

                file = mne.io.read_raw_edf(file_path)
                sampling_rate = file.info['sfreq']
                signal_labels = file.ch_names

                # Obtener la hora inicial
                start_time_edf = file.info['meas_date']  # UTC
                print(start_time_edf)

                data, times = file[:, :]
                # print(data.shape)
                unix_times =  [(start_time_edf + datetime.timedelta(seconds=t)).isoformat() for t in times]
                if debug:
                    end_time = datetime.datetime.now()
                    execution_time = end_time - start_time
                    print(bcolors.OKCYAN+"Done"+bcolors.ENDC+". ({} seconds)\n".format(execution_time.total_seconds()))
                    print('\nGenerando los JSON...')
                    start_time = datetime.datetime.now()
                row = subject_info[subject_info["Case"] == patient]
                gender = row['Gender']
                age = row['Age (years)']
                tags = {
                    "patient": patient,
                    'gender': gender.values[0], 
                    'age': age.values[0],
                    "sampling_rate": sampling_rate,
                }

                packed_data = split_data(data, unix_times, signal_labels, tags,  data.shape[1]//4)
                if debug:
                    end_time = datetime.datetime.now()
                    execution_time = end_time - start_time
                    print(bcolors.OKCYAN+"Done"+bcolors.ENDC+". ({} seconds)\n".format(execution_time.total_seconds()))
                # submatrices = np.array_split(data, 1000, axis=1)
                # subtimestamps = np.array_split(unix_times, 1000)
                # print(submatrices.shape)

                # for sm, sts in zip(submatrices, subtimestamps):
                #     for ch_data, ch_name in zip(sm, signal_labels):
                #         channels[ch_name] = ch_data.tolist()
                #         data_json = {
                #             'tags': tags,
                #             'timestamps': subtimestamps.tolist(),
                #             'channels' : channels
                #         }
                if debug:
                        print('\nEnviando MENSAJES...')
                        start_time_msgs = datetime.datetime.now()
                for message in packed_data:
                    # Envia el mensaje de Kafka
                    if debug:
                        print('\nEnviando mensaje...')
                        start_time = datetime.datetime.now()
                        print(data.shape[1]//4)
                        print(json.dumps(message))
                    producer.produce("chb-mit", json.dumps(message).encode('utf-8'))
                    producer.flush()
                    if debug:
                        end_time = datetime.datetime.now()
                        execution_time = end_time - start_time
                        print(bcolors.OKCYAN+"Done"+bcolors.ENDC+". ({} seconds)\n".format(execution_time.total_seconds()))
                if debug:
                    end_time = datetime.datetime.now()
                    execution_time = end_time - start_time_msgs
                    print(bcolors.OKCYAN+"Done MENSAJES"+bcolors.ENDC+". ({} seconds)\n".format(execution_time.total_seconds()))
                
    

                

end_time = datetime.datetime.now()
execution_time = end_time - start_time
print(bcolors.OKCYAN+"Done"+bcolors.ENDC+". ({} seconds)\n".format(execution_time.total_seconds()))

