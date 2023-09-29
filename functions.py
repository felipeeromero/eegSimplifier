import numpy as np
import math

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
