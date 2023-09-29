import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as fx
import math
import argparse

# For dealing with the EDF files
import mne
import datetime
import pytz



from functions import *

# Print the program title
print('====================')
print('\033[96m \033[1m EDF simplifier \033[0m')
print('====================')

# Define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-f","--frequency", help="Target frecuency in Hz [1]", default=1)
parser.add_argument("-p","--path", help="Path to the edf file")
parser.add_argument("-n","--name", help="Name of new edf file [synthetic_eeg]", default='synthetic_eeg')
parser.add_argument("-d","--device", help="Device where the functions are computed (gpu/cpu/both) [gpu]", default='gpu')
parser.add_argument("-F","--listFreqs",  nargs='+', help="List of frequencies to analize over time", default=[0, 50, 100, 150])
# parser.add_argument("-t","--test", action="store_true", help="Test with different window frequencies to measure times")
# parser.add_argument("-pl","--plot", action="store_true", help="Plot, show and save figures")
# parser.add_argument("-l","--latex", action="store_true", help="Configure the plot in Latex Fonts for a fancy result")

# Parse the command line arguments
args = parser.parse_args()

# Get the values for the command line arguments
target_rate = float(args.frequency)
freqs = [int(f) for f in args.listFreqs]
path = args.path
device = args.device
output_name = args.name
# test = args.test
# plot = args.plot

# Figures in LaTeX
# if args.latex:
#     plt.rcParams.update({
#         "text.usetex": True,
#         "font.family": "serif",
#         "font.serif": ["Palatino"],
#         "font.size": 16,
#     })

if device != 'cpu':
    # PyCUDA for GPU implementation
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda import gpuarray
    from pycuda.compiler import SourceModule

# GPU calling functions
def compute_windowed_fft_gpu(input_data, freq_m, freq_obj, freqs = [50, 100, 150]):
    num_samples = input_data.shape[1]
    num_var = input_data.shape[0]
    window_size = round(freq_m/freq_obj)
    output_data_size = math.ceil(num_samples / window_size) * num_var * len(freqs)
    output_data = np.empty(output_data_size, dtype=np.float32)
    print("Output size: {},".format(len(output_data)))

    # Copy the input data to the GPU
    d_input_data = gpuarray.to_gpu(input_data.flatten().astype(np.float32))
    d_freqs = gpuarray.to_gpu(np.array(freqs).astype(np.int32))
    # Create an array for the output data from de GPU
    d_output_data = gpuarray.empty((output_data_size), dtype=np.float32)

    # Configurar la ejecución del kernel
    block_size = (64,4)
    grid_size = (math.ceil(num_samples/window_size + block_size[0] - 1) // block_size[0], math.ceil(num_var+ block_size[1] - 1) // block_size[1])
    print("Grid size:", grid_size)
    fft_kernel(d_input_data, d_output_data, np.int32(num_samples), np.int32(num_var), np.int32(freq_m), d_freqs, np.int32(len(freqs)),
                          block=(block_size[0], block_size[1], 1), grid=(grid_size[0], grid_size[1]))

    # Copiar los resultados de vuelta a la CPU
    d_output_data.get(output_data)
    if (freq_m/window_size != freq_obj): print(bcolors.WARNING+"WARNING"+ bcolors.ENDC +"The computed output frequency is {} Hz".format(freq_m/window_size))
    return output_data

def resample_cpu(input_data, freq_m, freq_obj):
    num_samples = input_data.shape[1]
    num_var = input_data.shape[0]
    window_size = round(freq_m/freq_obj)
    output_data_size = math.ceil(num_samples / window_size) * 2 * num_var
    output_data = np.empty(output_data_size, dtype=np.float32)
    print("Output size: {},".format(len(output_data)))

    # Copiar los datos de entrada a la GPU
    d_input_data = gpuarray.to_gpu(input_data.flatten().astype(np.float32))
    # Crear un array para los datos de salida en la GPU
    d_output_data = gpuarray.empty((output_data_size), dtype=np.float32)

    # Configurar la ejecución del kernel
    block_size = (64,4)
    grid_size = (math.ceil(num_samples/window_size + block_size[0] - 1) // block_size[0], math.ceil(num_var+ block_size[1] - 1) // block_size[1])
    print(grid_size)
    preprocessData_kernel(d_input_data, d_output_data, np.int32(num_samples), np.int32(num_var),
                          block=(block_size[0], block_size[1], 1), grid=(grid_size[0], grid_size[1]))

    # Copiar los resultados de vuelta a la CPU
    d_output_data.get(output_data)
    if (freq_m/window_size != freq_obj): print(bcolors.WARNING+"WARNING"+ bcolors.ENDC +"The computed output frequency is {} Hz".format(freq_m/window_size))
    return output_data

print('\nReading EDF file...')
start_time = datetime.datetime.now()
file = mne.io.read_raw_edf(path)
sampling_rate = file.info['sfreq']
signal_labels = file.ch_names

# Obtener la hora inicial
local_time = file.info['meas_date']  # fecha y hora local
print(local_time)
local_timezone = pytz.timezone('Europe/Madrid')  # zona horaria local

# Convertir la hora local a UTC
utc_timezone = pytz.utc  # zona horaria UTC
start_time_edf = local_time.astimezone(local_timezone)
print(start_time_edf)
end_time = datetime.datetime.now()
execution_time = end_time - start_time
print(bcolors.OKCYAN+"Done"+bcolors.ENDC+". ({} seconds)\n".format(execution_time.total_seconds()))


if (device!='cpu'):
    print('Compiling kernel code...')
    start_time = datetime.datetime.now()
    with open("gpu_functions.cu", "r") as f:
        kernel_code = f.read()
        kernel_code = kernel_code.replace("winSize = 512;", "winSize = {};".format(round(sampling_rate/target_rate)))

    # Compilar el kernel CUDA
    module = SourceModule(kernel_code)
    resampling_kernel = module.get_function("resample")
    fft_kernel = module.get_function("compute_windowed_fft")
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    print(bcolors.OKCYAN+"Done"+bcolors.ENDC+". ({} seconds)\n".format(execution_time.total_seconds()))



# print(signal_labels)
print('Obtaining annotations and timestamps...')
start_time = datetime.datetime.now()
data, times = file[:, :]
reduced_times = reduce_time_size(times, sampling_rate, target_rate)
unix_times =  [(start_time_edf + datetime.timedelta(seconds=t)).isoformat() for t in reduced_times]


annotations = file.annotations
onset_times = annotations.onset
unix_times_an = [(start_time_edf + datetime.timedelta(seconds=t)).isoformat() for t in onset_times]
end_time = datetime.datetime.now()
execution_time = end_time - start_time
print(bcolors.OKCYAN+"Done"+bcolors.ENDC+". ({} seconds)\n".format(execution_time.total_seconds()))

del file

if (device != 'gpu'):
    print('\nComputing in CPU...')
    start_time = datetime.datetime.now()
    output_data = compute_windowed_fft_cpu(data, sampling_rate, target_rate, freqs=freqs)
    # output_data_cpu = preprocess_data_cpu(data, sampling_rate, target_rate)
    end_time = datetime.datetime.now()
    execution_time_cpu = end_time - start_time
    print("Elapsed time ({}CPU{}):".format(bcolors.FAIL, bcolors.ENDC), execution_time_cpu.total_seconds(), "seconds")

if (device != 'cpu'):
    print('\nComputing in GPU...')
    start_time = datetime.datetime.now()
    output_data = compute_windowed_fft_gpu(data, sampling_rate, target_rate, freqs=freqs)
    # output_data = preprocess_data_cpu(data, sampling_rate, target_rate)
    end_time = datetime.datetime.now()
    execution_time_gpu = end_time - start_time
    print("Elapsed time ({}GPU{}):".format(bcolors.OKGREEN, bcolors.ENDC), execution_time_gpu.total_seconds(), "seconds")

if (device != 'cpu' and device != 'gpu'):
    speedup = execution_time_cpu/execution_time_gpu
    print("Speedup: {}{}{}".format(bcolors.OKCYAN, speedup, bcolors.ENDC)) 

new_ch_names = []
new_data = []
num_freqs = len(freqs)
for var, lab in enumerate(signal_labels):
    for i, f in enumerate(freqs):
        new_ch_names.append(lab+'_{}'.format(f))
        new_data.append(output_data[var*math.ceil(len(output_data)/(data.shape[0]))+i:(var+1)*math.ceil(len(output_data)/(data.shape[0])):num_freqs])

info = mne.create_info(ch_names=new_ch_names, sfreq=target_rate)
output_data /= sampling_rate
# Create an MNE Raw object
raw = mne.io.RawArray(new_data, info)
raw.set_meas_date(local_time)
raw.set_annotations(annotations)

output_path = "./output/"+output_name+".edf"
# Save the EEG data as an EDF file
# raw.save(output_path, overwrite=True)
mne.export.export_raw(output_path, raw, overwrite=True)

print(f'Saved EEG data to {output_path}')