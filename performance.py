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

# PyCUDA for GPU implementation
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule

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

# Print the program title
print('=================================')
print('\033[96m \033[1m Time performance Analyzer \033[0m')
print('=================================')

# Define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-p","--path", help="Path to the edf file")
parser.add_argument("-n","--number", help="Number of measurements to be averaged", default=3)
parser.add_argument("-pl","--plot", action="store_true", help="Plot, show and save figures")
parser.add_argument("-l","--latex", action="store_true", help="Configure the plot in Latex Fonts for a fancy result")

# Parse the command line arguments
args = parser.parse_args()

# Get the values for the command line arguments

path = args.path
number = args.number


plot = args.plot

# Figures in LaTeX
if args.latex:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
        "font.size": 16,
    })

print('\nReading EDF file...')
start_time = datetime.datetime.now()
file = mne.io.read_raw_edf(path)
sampling_rate = file.info['sfreq']
# signal_labels = file.ch_names
data, _ = file[:, :]

end_time = datetime.datetime.now()
execution_time = end_time - start_time
print(bcolors.OKCYAN+"Done"+bcolors.ENDC+". ({} seconds)\n".format(execution_time.total_seconds()))


target_rates = [0.25, 0.5, 1, 2, 4, 8]
kernel_codes = []

# Cargar el archivo de código fuente del kernel CUDA
with open("gpu_functions.cu", "r") as f:
    kernel_code = f.read()

for tr in target_rates: kernel_codes.append(kernel_code.replace("winSize = 512;", "winSize = {};".format(round(sampling_rate/tr))))

# Inicializar GPU
print('\nInitializing GPU...')
start_time = datetime.datetime.now()
module = SourceModule(kernel_code)
fft_kernel = module.get_function("compute_windowed_fft")
end_time = datetime.datetime.now()
compute_windowed_fft_gpu(data, sampling_rate, target_rates[-1], [5, 20, 30])
execution_time = end_time - start_time
print(bcolors.OKCYAN+"Done"+bcolors.ENDC+". ({} seconds)\n".format(execution_time.total_seconds()))

freqs = [50, 100, 150, 200]

ex_times_cpu = []
ex_times_gpu = []
speed_up = []
for i in range(number):
    print('\nMeasurement #{}'.format(i+1))
    print("------------------------------")
    for f,kc in zip(target_rates, kernel_codes):

        print('\nCompiling kernel code for the frequency {} Hz...'.format(f))
        start_time = datetime.datetime.now()
        module = SourceModule(kc)
        fft_kernel = module.get_function("compute_windowed_fft")
        end_time = datetime.datetime.now()
        execution_time = end_time - start_time
        print(bcolors.OKCYAN+"Done"+bcolors.ENDC+". ({} seconds)".format(execution_time.total_seconds()))

        print('\nComputing in CPU...')
        start_time = datetime.datetime.now()
        output_data_cpu = compute_windowed_fft_cpu(data, sampling_rate, f, freqs=freqs)
        end_time = datetime.datetime.now()
        time_cpu = end_time - start_time
        ex_times_cpu.append(time_cpu.total_seconds()*1e3)


        print('\nComputing in GPU...')
        start_time = datetime.datetime.now()
        output_data = compute_windowed_fft_gpu(data, sampling_rate, f, freqs=freqs)
        end_time = datetime.datetime.now()
        time_gpu = end_time - start_time
        ex_times_gpu.append(time_gpu.total_seconds()*1e3)

        s_up = time_cpu/time_gpu
        speed_up.append(s_up)
        print("[{} Hz]:".format(f) ," Elapsed time ({}CPU{}):".format(bcolors.FAIL, bcolors.ENDC), time_cpu.total_seconds(), "seconds")
        print("[{} Hz]:".format(f) ," Elapsed time ({}GPU{}):".format(bcolors.OKGREEN, bcolors.ENDC), time_gpu.total_seconds(), "seconds")
        print("[{} Hz]:".format(f)," {}Speedup{}: {}".format(bcolors.OKCYAN, round(s_up,2), bcolors.ENDC)) 

if plot:
    print('\nDrawing plots...')
    start_time = datetime.datetime.now()
    times_cpu_m = np.reshape(ex_times_cpu,(3,6))
    times_gpu_m = np.reshape(ex_times_gpu,(3,6))
    speed_up_m = np.reshape(speed_up,(3,6))

    times_cpu_mean = np.mean(times_cpu_m, axis=0)
    times_gpu_mean = np.mean(times_gpu_m, axis=0)
    speed_up_mean = np.mean(speed_up_m, axis=0)
    times_cpu_std = np.std(times_cpu_m, axis=0)
    times_gpu_std = np.std(times_gpu_m, axis=0)
    speed_up_std = np.std(speed_up_m, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(np.arange(6)-0.2, times_cpu_mean*1e-3, 0.35, zorder=10,label='CPU')
    ax.bar(np.arange(6)+0.2, times_gpu_mean*1e-3, 0.35, zorder=10,label='GPU')
    ax.grid()
    ax.legend()
    ax.set_xlabel("Window Rate")
    ax.set_ylabel("Elapsed time (s)")
    ax.set_xticks(np.arange(6),["0.25 Hz", "0.5 Hz", "1 Hz", "2 Hz", "4 Hz", "8Hz"])
    ax.set_ylim([0, 160])
    fig.savefig("img/cpu_vs_gpu.eps",  bbox_inches='tight')
    fig.savefig("img/cpu_vs_gpu.png", bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(np.arange(6), speed_up_mean, 0.35, zorder=10,label='Speedup')
    ax.errorbar(np.arange(6), speed_up_mean, speed_up_std, color='0.0', fmt='none', capsize=16, zorder=11)
    ax.grid()
    # ax.legend()
    ax.set_xlabel("Window Rate")
    ax.set_ylabel("Speedup")
    ax.set_xticks(np.arange(6),["0.25 Hz", "0.5 Hz", "1 Hz", "2 Hz", "4 Hz", "8Hz"])
    fig.savefig("img/speed_up.eps",  bbox_inches='tight')
    fig.savefig("img/speed_up.png", bbox_inches='tight', dpi=300)
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    print(bcolors.OKCYAN+"Done"+bcolors.ENDC+". ({} seconds)".format(execution_time.total_seconds()))
