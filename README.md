# Accelerating Epilepsy Diagnosis and Prediction on EEG Signals with GPU-Based Frequency Domain Analysis and Resampling.

## 1. Introduction

The importance of precise seizure detection in epilepsy is emphasized in this work. Current diagnostic methods, such as Video-Electroencephalography, are often costly and time-consuming. However, recent research demonstrates the potential of artificial intelligence for seizure detection using EEG or ECG data. To address challenges related to high-frequency data acquisition, this repository proposes the utilization of Fast Fourier Transform for frequency analysis and GPUs for efficient data processing, enabling reduced sampling frequencies. These advancements enhance data storage, transmission, and research efficiency, thereby facilitating neural network training for epilepsy prediction and real-time execution.

This repository contains the functions implemented to obtain the results in the article with the same title. The primary objective of this work has been to develop an application that reduces the file size analyzed by physicians for the detection of epileptic seizures. This size reduction allows for storage in a database, such as InfluxDB. Additionally, the windowed FFT has provided a more intuitive way to graphically detect seizures, simplifying the work for physicians and enabling neural networks to predict these anomalies based on simplified data.

## Dependencies
To execute the code, it is necessary to install the following libraries:
* MNE: An open-source Python package for exploring, visualizing, and analyzing human neurophysiological data. (`pip install mne`)
* PyCUDA: A package for accessing Nvidia’s CUDA parallel computation API from Python. It is possible to run the code without a GPU and this library if the argument `--device` (`-d`) is set to "cpu" (`pip install pycuda`) 
* EDFlib: A programming library to read/write EDF+ (16-bit) and BDF+ (24-bit) files, written in pure Python.(`pip install EDFlib-Python`)
* Datetime libraries: (`pip install datetime pytz`)
* Other required libraries: (`pip install numpy matplotlib`)

## 2. Usage
The program `edf_simplfier.py` takes an EDF file obtained from EEG measurements and converts it into another file containing the amplitudes of the desired frequencies (by default, 0, 50, 100, and 150 Hz) for each of the signals. Therefore, it reduces the number of measurements per second in exchange for increasing the number of channels. For example, for the signal marked as 'T1', you have 'T1_0', 'T1_50', 'T1_100', and 'T1_150' in the new file.

### 2.1 Quick test
To test the program, an example EDF file (`example.edf`) containing annotations by physicians during epileptic seizures has been provided. If you have PyCUDA installed and the corresponding drivers for an NVIDIA card, you can try:

```
py edf_simplifier.py -p "./example.edf"
```
If you lack an NVIDIA card or were unable to install the library, run:
```
py edf_simplifier.py -p "./example.edf" -d "cpu"
```

In both cases, an output file named `synthetic_eeg.edf` should have been created in the `output` folder.

### 2.2. Personalization
It is possible to change the configuration of the output EDF file by modifying the following arguments:
* Path of the EDF file (`-p`/`--path`): The path containing the EDF file to analyze.
* Target frequency (`-f`/`--frequency`): The frequency at which the FFT is performed. Defaults to 1.
* Name of the output EDF file (`-n`/`--name`). Defaults to "synthetic_eeg".
* Device (`-d`/`--device`): The device where the functions are computed ("gpu"/"cpu"/"both"). Defaults to "gpu".
* List of frequencies (`-F`/`--listFreqs`): A list of frequencies whose amplitudes are analyzed over time. Defaults to [0, 50, 100, 150].

#### Example
Command to generate a new EDF file called `new_eeg.edf` with a frequency of 2 data points per variable and second, using both CPU and GPU (for measuring the speedup), for the frequencies 40, 50, and 75 Hz from the `example.edf` file:
```
py edf_simplifier.py -p "./example.edf" -f 2 -n "new_eeg" -d "both" -F 40 50 75
```

## 3. Visualization
You can visualize the generated file with an [online viewer](https://bilalzonjy.github.io/EDFViewer/EDFViewer.html) for this type of file.

## License
This software is provided 'as is', with the best intention but without any kind of warranty and responsibility for stability or correctness. It can be freely used, distributed, and modified by anyone interested, but a reference to this source is required.

## Citing
To cite this software, please, cite the associated paper as follows: L. Felipe Romero, Marcos Lupión, Luis F. Romero, J.F. Sanjuan, and Pilar M. Ortigosa. Accelerating Epilepsy Diagnosis and Prediction on EEG Signals with GPU-Based Frequency Domain Analysis and Resampling

. _Journal of Supercomputing_, 2023.

## Contact
For any questions or suggestions, feel free to contact:

- L. Felipe Romero: [fr@uma.es](fr@uma.es)
- Marcos Lupión: [marcoslupion@ual.es](marcoslupion@ual.es)