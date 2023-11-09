import numpy as np
import math
import re
import os
import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from pycuda import gpuarray

def digitalize(sensor, value):
    """
    Digitalize a sensor reading based on the sensor type.
    
    Args:
        sensor (str): The sensor type.
        value (float): The sensor reading.
    
    Returns:
        uint8: The digitalized value.
    """
    Vcc = 3
    if "SpO2" not in sensor: 
        sensor = sensor[:-1]
    match sensor:
        case "ECG":
            return ((value*1.019/Vcc+0.5)*2**8).astype(np.uint8)
        case "EDA":
            return (value*0.12/Vcc*2**8).astype(np.uint8)
        case "TEMP":
            return value
        case "EMG":
            return ((value/Vcc+0.5)*2**8).astype(np.uint8)
        case "SpO23":
            return (value*1.2*2**8*1e-3).astype(np.uint8)
        case "SpO24":
            return (value*1.2*2**8*1e-3).astype(np.uint8)
        case "%SpO25":
            return value

def reduce_time_size(times, freq_m, freq_obj):
    """
    Reduce the time array size based on frequency.

    Args:
        times (ndarray): The input time array.
        freq_m (float): Sampling frequency of the original data.
        freq_obj (float): Target frequency for reduction.

    Returns:
        ndarray: The reduced time array.
    """
    indices = np.arange(0, len(times), int((freq_m // freq_obj)))
    return times[indices]

def compute_windowed_fft_cpu(input_data, freq_m, freq_obj, freqs=[50, 100, 150]):
    """
    Compute the windowed FFT of input data.

    Args:
        input_data (ndarray): Input data array.
        freq_m (float): Sampling frequency of the input data.
        freq_obj (float): Target frequency for FFT computation.
        freqs (list): List of frequencies for FFT computation.

    Returns:
        ndarray: The computed FFT data.
    """
    num_samples = input_data.shape[1]
    num_var = input_data.shape[0]
    window_size = round(freq_m / freq_obj)
    num_freqs = len(freqs)
    output_data_size = math.ceil(num_samples / window_size) * num_var * num_freqs
    output_data = np.empty(output_data_size, dtype=np.float32)

    for v in range(num_var):
        output_offset = math.ceil((v * num_samples) / window_size * num_freqs)
        for i in range(0, num_samples, window_size):
            fft_r = np.fft.fft(input_data[v, i:i+window_size])
            ind_freqs = np.round(np.array(freqs) * (window_size // freq_m)).astype(int)
            output_index = output_offset + i // window_size * num_freqs
            for j in range(0, num_freqs):
                try:
                    output_data[output_index + j] = np.abs(fft_r[ind_freqs[j]])
                except: 
                    output_data[output_index + j] = 0

    return output_data

def resample_cpu(input_data, freq_m, freq_obj):
    """
    Resample input data based on frequencies.

    Args:
        input_data (ndarray): Input data array.
        freq_m (float): Sampling frequency of the input data.
        freq_obj (float): Target frequency for resampling.

    Returns:
        ndarray: The resampled data.
    """
    num_samples = input_data.shape[1]
    num_var = input_data.shape[0]
    output_data_size = math.ceil(num_samples * freq_obj / freq_m * 2 * num_var)
    output_data = np.empty(output_data_size, dtype=np.float32)

    for v in range(num_var):
        output_offset = int(math.ceil(v * num_samples * freq_obj / freq_m * 2))

        for i in range(0, num_samples, int(freq_m / freq_obj)):
            avg = np.mean(input_data[v, i:i+int(freq_m/freq_obj)])
            max_val = np.max(input_data[v, i:i+int(freq_m/freq_obj)])
            min_val = np.min(input_data[v, i:i+int(freq_m/freq_obj)])
            scale = max(abs(max_val - avg), abs(min_val - avg))

            output_index = i * int(freq_obj/freq_m)
            output_data[output_offset + output_index * 2] = avg
            output_data[output_offset + output_index * 2 + 1] = scale

    return output_data

# GPU calling functions
def compute_gpu(kernel, input_data, freq_m, freq_obj, freqs = None):
    num_samples = input_data.shape[1]
    num_var = input_data.shape[0]
    data_per_window = 2 if freqs is None else len(freqs)
    window_size = round(freq_m/freq_obj)

    print(data_per_window)

    output_data_size = math.ceil(num_samples / window_size) * num_var * data_per_window
    output_data = np.empty(output_data_size, dtype=np.float32)
    print("Output size: {},".format(len(output_data)),"Input: {} ({}x{}),".format(num_samples*num_var,num_samples,num_var))

    # Copy the input data to the GPU
    d_input_data = gpuarray.to_gpu(input_data.flatten().astype(np.float32))
    d_freqs = gpuarray.to_gpu(np.array(freqs).astype(np.int32))
    # Create an array for the output data from de GPU
    d_output_data = gpuarray.empty((output_data_size), dtype=np.float32)

    # Configurar la ejecución del kernel
    block_size = (32,8,4)
    grid_size = (math.ceil(num_samples/window_size + block_size[0] - 1) // block_size[0], math.ceil(num_var+ block_size[1] - 1) // block_size[1], 1)
    print("Configuration: <<<{},{}>>>".format(grid_size, block_size))
    kernel(d_input_data, d_output_data, np.int32(num_samples), np.int32(num_var), np.int32(freq_m), d_freqs, np.int32(data_per_window),
                          block=(block_size[0], block_size[1], block_size[2]), grid=(grid_size[0], grid_size[1], grid_size[2]))

    # Copiar los resultados de vuelta a la CPU
    d_output_data.get(output_data)
    # print(output_data)
    if (freq_m/window_size != freq_obj): print(bcolors.WARNING+"WARNING"+ bcolors.ENDC +"The computed output frequency is {} Hz".format(freq_m/window_size))
    return output_data

class bcolors:
    """Format the console output
    """    
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def extract_session_info(session_text, sessions = {}):
  """Extracts the sessions from the given text.

  Args:
    text: A string containing the text of the sessions.

  Returns:
    A list of dictionaries containing the information about each session.
  """

  # Extract the session information.
  file_name = re.search(r'File Name: (.+)\.edf', session_text).group(1)
  n_seizures_match = re.search(r'Number of Seizures in File: (\d+)', session_text)
  n_seizures = int(n_seizures_match.group(1))
  line_start = n_seizures_match.end()+1
  lines = session_text[line_start:].split('\n')
  sessions[file_name] = []
  for i in range(n_seizures):
    seizure_info = {}
    seizure_info['seizure_number'] = i+1
    seizure_info['seizure_start_time'] = int(lines[2*i].split(' ')[-2])
    seizure_info['seizure_end_time'] = int(lines[2*i+1].split(' ')[-2])
    sessions[file_name].append(seizure_info)



  return sessions

def extract_summary(filename):
  """Extract the information for each session in the text file.

  Args:
      filename: The file name.

  Return:
      A dict with the seizures of each session of the patient
  """

  # Extract the information for each session in the text file.
  sessions = {}
  with open(filename, 'r') as f:
    text = f.read()
    for i, session_text in enumerate(text.split('\n\n')):
      if i > 1 and session_text.startswith('File'):
        # print(session_text)
        # session_info = extract_session_info(session_text)
        sessions = extract_session_info(session_text, sessions)
    f.close()

  return sessions

def escribir_a_influx(data, bucket = 'raw', url = 'http://localhost:8086', org = 'ual'):
    token = os.environ.get("INFLUX_TOKEN")
    # url = "http://150.214.150.187:8086"



    write_client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
    write_api = write_client.write_api(write_options=SYNCHRONOUS)

    data_t = np.array(data).transpose()
    data_to_store = []
    row = subject_info[subject_info["Case"] == patient]
    gender = row['Gender']
    age = row['Age (years)']
    tags = {'patient': patient, 'gender': gender.values[0], 'age': age.values[0]}

    for i, registro in enumerate(data_t):
        # Almacenar los datos en un objeto con los tags específicos
        fields = {}
        for d, lab in zip(registro, signal_labels):
            fields[lab]=d

        data_point = {
            "measurement": "chb-mit",
            "tags": tags,
            "time": unix_times[i],
            "fields": fields
        }

        data_to_store.append(data_point)

        # Escribir en InfluxDB si hay 10000 elementos en data_to_store
        if len(data_to_store) == 10000:
            write_api.write(bucket, org, data_to_store)
            data_to_store = []

    # Escribir en InfluxDB cualquier elemento restante en data_to_store
    if len(data_to_store) > 0:
        write_api.write(bucket, org, data_to_store)

def split_data(matrix, timestamps, channel_names, tags, num_divisions=1000):
  """Divide una matriz en submatrices, agrupando los datos por canal y timestamp.

  Args:
    matrix: La matriz a dividir.
    num_divisions: El número de submatrices en las que se dividirá la matriz.

  Return:
    Una lista de objetos JSON, cada uno de los cuales contiene una submatriz de datos para cada canal y timestamp.
"""

  # Calcula el tamaño de cada submatriz.
  submatrix_size = math.ceil(matrix.shape[1] / num_divisions)

  # Crea una lista de submatrices.
  packed_data = []
  for i in range(num_divisions):
    start_index = i * submatrix_size
    end_index = (i + 1) * submatrix_size
    submatrix = matrix[:, start_index:end_index]
    subtimestamps = timestamps[start_index:end_index]
    channels = {}
    for ch_data, ch_name in zip(submatrix, channel_names):
        channels[ch_name] = ch_data.tolist()
    data_json = {
        'tags': tags,
        'timestamps': subtimestamps,
        'channels' : channels
    }
    packed_data.append(data_json)


  return packed_data

def normalize_data(data):
  """
  Normaliza los datos flotante a entero entre 0 y 255 (int8) teniendo cuidado con los valores atípicos.

  Args:
    data: Los datos flotante a normalizar.

  Returns:
    Los datos normalizados en formato int8.
  """

  minimos_por_canal = np.min(data, axis=1, keepdims=True)  # Añade keepdims=True para mantener la forma (23, 1)
  maximos_por_canal = np.max(data, axis=1, keepdims=True)
  # l2_norm = np.linalg.norm(data, 2, axis=1, keepdims=False)
  
  norm_data = np.rint((data - minimos_por_canal) / (maximos_por_canal - minimos_por_canal) * 255)

  # print(norm_data)
  # Convertir los datos a tipo int8
  norm_data = norm_data.astype(np.uint8)
  # print(norm_data2-norm_data)

  return norm_data