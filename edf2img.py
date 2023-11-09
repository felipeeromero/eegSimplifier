import numpy as np
import math
import argparse
import os
import pandas as pd
import sys
from PIL import Image


# For dealing with the EDF files
import mne
import datetime
import pytz
import json

from functions import *

# Define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-f","--frequency", help="Target frecuency in Hz [1]", default=1)
parser.add_argument("-p","--path", help="Path to the edf file", default='Y:\datasets\chb-mit')
parser.add_argument("-o","--output", help="Output path", default='Y:\datasets\chb-mit-img')
parser.add_argument("-d","--device", help="Device where the functions are computed (gpu/cpu/both) [gpu]", default='gpu')
parser.add_argument("-F","--listFreqs",  nargs='+', help="List of frequencies to analize over time", default=[16, 32, 48])
parser.add_argument("-l","--labelled", action="store_true", help="Save the figure with its label")

# parser.add_argument("-t","--test", action="store_true", help="Test with different window frequencies to measure times")
# parser.add_argument("-pl","--plot", action="store_true", help="Plot, show and save figures")
# parser.add_argument("-l","--latex", action="store_true", help="Configure the plot in Latex Fonts for a fancy result")

# Parse the command line arguments
args = parser.parse_args()
debug = getattr(sys, 'gettrace', None)()

target_rate = float(args.frequency)
freqs = [int(f) for f in args.listFreqs]
dataset_path = args.path
device = args.device
output_path = args.output
labelled = args.labelled

if device != 'cpu':
    # PyCUDA for GPU implementation
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda import gpuarray
    from pycuda.compiler import SourceModule

# Print the program title
print('==================')
print('\033[96m \033[1m EDF 2 Img \033[0m')
print('==================')

sampling_rate = 256
window_size = round(sampling_rate/target_rate)

# Path to the dataset
with open("gpu_functions.cu", "r") as f:
    kernel_code = f.read()
kernel_code = kernel_code.replace("winSize = 512;", "winSize = {};".format(window_size))


# Compilar el kernel CUDA
module = SourceModule(kernel_code)
fft_kernel = module.get_function("compute_windowed_fft")

subject_info = pd.read_csv(dataset_path+"\SUBJECT-INFO", delimiter='\t')


print('\nReading EDF files...')
start_time = datetime.datetime.now()

for patient in os.listdir(dataset_path):
    patient_path = os.path.join(dataset_path, patient)
    output_patient_path = os.path.join(output_path, patient)

    if os.path.isdir(patient_path):
        if not os.path.exists(output_patient_path):
            os.makedirs(output_patient_path)
        sessions = extract_summary(os.path.join(patient_path, patient)+"-summary.txt")
        for file_name in os.listdir(patient_path):
            if file_name.endswith('.edf'):                   
                file_path = os.path.join(patient_path, file_name)

                file = mne.io.read_raw_edf(file_path, verbose='ERROR')
                if sampling_rate != file.info['sfreq']:
                    with open("gpu_functions.cu", "r") as f:
                        kernel_code = f.read()
                    kernel_code = kernel_code.replace("winSize = 512;", "winSize = {};".format(round(sampling_rate/target_rate)))
                    sampling_rate = file.info['sfreq']
                    window_size = round(sampling_rate/target_rate)

                signal_labels = file.ch_names
                sorted_channel_indices = sorted(range(len(signal_labels)), key=lambda k: signal_labels[k])
                sorted_channel_names = [signal_labels[i] for i in sorted_channel_indices]

                # Reordenar los canales en el Raw object
                raw_sorted = file.copy().reorder_channels(sorted_channel_names)

                # Obtener la hora inicial
                start_time_edf = file.info['meas_date']  # UTC
                

                data, times = raw_sorted[:, :]

                if (device == 'cpu'):
                    print('\nComputing in CPU...')
                    start_time = datetime.datetime.now()
                    output_data = compute_windowed_fft_cpu(data, sampling_rate, target_rate, freqs=freqs)
                    end_time = datetime.datetime.now()
                    execution_time_cpu = end_time - start_time
                    print("Elapsed time ({}CPU{}):".format(bcolors.FAIL, bcolors.ENDC), execution_time_cpu.total_seconds(), "seconds")

                else:
                    print('\nComputing in GPU...')
                    start_time = datetime.datetime.now()
                    output_data = compute_gpu(fft_kernel, data, sampling_rate, target_rate, freqs=freqs)
                    end_time = datetime.datetime.now()
                    execution_time_gpu = end_time - start_time
                    print("Elapsed time ({}GPU{}):".format(bcolors.OKGREEN, bcolors.ENDC), execution_time_gpu.total_seconds(), "seconds")   


                row = subject_info[subject_info["Case"] == patient]
                gender = row['Gender']
                age = row['Age (years)']

                reduced_times = reduce_time_size(times, sampling_rate, target_rate)
                is_crisis = np.zeros_like(reduced_times)
                try:
                    for crisis in sessions[file_name.replace('.edf','')]:
                        is_crisis += (reduced_times >= crisis['seizure_start_time']) & (reduced_times <= crisis['seizure_end_time'])
                    # print(labels_crisis)
                except:
                    print(f"{bcolors.WARNING}WARNING{bcolors.ENDC} This session was not annotated")


                print('\nConverting to images...')
                start_time = datetime.datetime.now()
                num_samples = data.shape[1]
                num_var = data.shape[0]
                reshaped_data = output_data.reshape((num_var, math.ceil(num_samples / window_size), len(freqs)))
                norm_data = normalize_data(reshaped_data).transpose((1, 0, 2))

                # Calcula el número de imágenes.
                num_images = int(len(output_data)/(num_var*num_var*3))

                images = norm_data[:num_images*num_var].reshape((num_images,num_var, num_var,3))
                labels_crisis = is_crisis[:num_images*num_var].reshape((num_images,num_var))
                label_pixs =(np.stack((labels_crisis, labels_crisis, labels_crisis), axis=-1)*255).astype(np.uint8)
                

                for i, (image, label) in enumerate(images, labels_crisis):
                    nombre_archivo = f"{output_patient_path}\{file_name.replace('.edf','')}_{gender.values[0]}_{int(age.values[0])}_{i}"
                    # Guardar la imagen
                    with open(f"{nombre_archivo}.npy", 'wb') as f:
                        np.save(f, label)
                  
                    new_image = np.append(image,  np.expand_dims(label_pixs[i], axis=1), axis=1)
                    # print(new_image.shape)
                    img_name = f"{nombre_archivo}.png"
                    if labelled:
                        Image.fromarray(new_image).transpose(Image.ROTATE_90).save(img_name)
                    else:
                        Image.fromarray(image).transpose(Image.ROTATE_90).save(img_name)
                    

                end_time = datetime.datetime.now()
                execution_time = end_time - start_time
                print(bcolors.OKCYAN+"Done"+bcolors.ENDC+". ({} seconds)\n".format(execution_time.total_seconds()))


                


