    #include <stdio.h>
    #include <math.h>
    // #include <cuComplex.h>
    // #include <cuda.h>
    // #include <cuda_runtime.h>
    
    const int winSize = 512; // Size of the FFT window


extern "C" {

    #define PI 3.14159265358979323846

    // Definition of a complex number structure
    typedef struct {
        float x;
        float y;
    } Complex;

    // Complex number multiplication
    __device__ Complex Complex_mul(Complex a, Complex b)
    {
        Complex c;
        c.x = a.x * b.x - a.y * b.y;
        c.y = a.x * b.y + a.y * b.x;
        return c;
    }

    // Complex number addition
    __device__ Complex Complex_add(Complex a, Complex b)
    {
        Complex c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        return c;
    }

    // Complex number subtraction
    __device__ Complex Complex_sub(Complex a, Complex b)
    {
        Complex c;
        c.x = a.x - b.x;
        c.y = a.y - b.y;
        return c;
    }

    // Fast Fourier Transform (FFT) algorithm
    __device__ void fft(Complex* x, unsigned int N)
    {
        // The code implements the FFT algorithm for complex numbers 'x' with size 'N'.
        // Implementation of the Cooley-Tukey FFT algorithm
        // (Decimation-in-time radix-2)
        // DFT
        unsigned int k = N, n;
        float thetaT = PI / N;
        Complex phiT = {cos(thetaT), -sin(thetaT)}, T;
        while (k > 1)
        {
            n = k;
            k >>= 1;
            phiT = Complex_mul(phiT, phiT);
            T = {1.0L, 0.0L};
            for (unsigned int l = 0; l < k; l++)
            {
                for (unsigned int a = l; a < N; a += n)
                {
                    unsigned int b = a + k;
                    Complex t = Complex_sub(x[a], x[b]);
                    x[a] = Complex_add(x[a], x[b]);
                    x[b] = Complex_mul(t, T);
                }
                T = Complex_mul(T, phiT);
            }
        }
        // Decimate
        unsigned int m = static_cast<unsigned int>(log2f(N));
        for (unsigned int a = 0; a < N; a++)
        {
            unsigned int b = a;
            // Reverse bits
            b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
            b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
            b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
            b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
            b = ((b >> 16) | (b << 16)) >> (32 - m);
            if (b > a)
            {
                Complex t = x[a];
                x[a] = x[b];
                x[b] = t;
            }
        }
        //// Normalize (This section make it not working correctly)
        //Complex f = Complex(1.0) / sqrt(N);
        //for (unsigned int i = 0; i < N; i++)
        //    x[i] *= f;
    }

    // Function to select a subarray from a larger array with zero padding
    __device__ void select_subarray(float array[], Complex *subarray, int first, int last, int i, int winSize) {
        // This function selects a subarray from 'array' and stores it in 'subarray'.

        int half = winSize / 2;

        for (int j = 0; j < winSize; j++) {
            int index = i - half + j;
            if (index < first || index >= last) {
                subarray[j].x = 0.0f;
                subarray[j].y = 0.0f;
            } else {
                float v = array[index];
                subarray[j].x = v;
                subarray[j].y = 0.0f;
            }
        }
}
    
    // Kernel for resampling and processing data
    __global__ void resample(float *inputData, float *outputData, int numSamples, int numVariables) {
        // This kernel resamples and processes data based on 'inputData' and stores results in 'outputData'.
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int var = blockIdx.y * blockDim.y + threadIdx.y;
        int stride_x = blockDim.x * gridDim.x * winSize;
        int stride_y = blockDim.y * gridDim.y;

        for (int v = var; v < numVariables; v += stride_y) {
            int outputOffset = static_cast<int>(ceil((float)numSamples / (float)winSize) * v * 2);
            for (int i = v * numSamples + idx * winSize; i < (v + 1) * numSamples; i += stride_x) {
                float sum = 0.0f;
                float maxVal = inputData[i];
                float minVal = inputData[i];
                int range = winSize/2;
                int startIdx = (i - range > v * numSamples) ? i - range : v * numSamples;
                int endIdx =  (i + range < (v + 1) * numSamples) ? i + range : (v + 1) * numSamples;

                int c = 0;

                for (int j = startIdx; j < endIdx; j++) {
                    float val = inputData[j];
                    sum += val;
                    maxVal = fmaxf(maxVal, val);
                    minVal = fminf(minVal, val);
                    c++;
                }

                float avg = sum / c;
                float scale = fmaxf(fabsf(maxVal - avg), fabsf(minVal - avg));


                // int outputIndex = static_cast<int>(round((i - v * numSamples) * (freq_obj / freq_m)));
                outputData[outputOffset + idx * 2] = avg;
                outputData[outputOffset + idx * 2 + 1] = scale;

            }
        }
    }

    // Kernel for creating an identity matrix from input data for testing
    __global__ void identityMatrix(float *inputData, float *outputData, int numSamples, int numVariables) {
        // This kernel creates an identity matrix from 'inputData' and stores it in 'outputData'.
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int var = blockIdx.y * blockDim.y + threadIdx.y;
        int stride_x = blockDim.x * gridDim.x;
        int stride_y = blockDim.y * gridDim.y;

        for (int v = var; v < numVariables; v += stride_y) {
            for (int i = idx ; i <  numSamples; i += stride_x) {
                
                outputData[v * numSamples + idx] = inputData[v * numSamples + idx];
            }
        }   
    }

    /**
     * @brief Compute Windowed Fast Fourier Transform (FFT)
     *
     * This CUDA kernel computes the windowed Fast Fourier Transform (FFT) of the input data 'inputData'
     * and stores the result in 'outputData'. The FFT is computed using a sliding window approach
     * on the input data.
     *
     * @param[in] inputData   Pointer to the input data to be processed.
     * @param[out] outputData Pointer to the output data where FFT results will be stored.
     * @param[in] numSamples  The total number of samples in the input data per variable.
     * @param[in] numVariables The number of variables or signals in the input data.
     * @param[in] freq_m      The reference frequency used for FFT computation.
     * @param[in] freqs       An array of integer frequencies to compute FFT at.
     * @param[in] numFreqs    The number of frequencies in the 'freqs' array.
     *
     * The function is executed as a CUDA kernel where each thread processes a portion of the data.
     * The parameters 'blockIdx', 'blockDim', and 'threadIdx' are used to determine the thread's
     * position within the grid of threads.
     *
     * The kernel operates as follows:
     * 1. Each thread calculates its 'idx' (index of the window) and 'var' (variable) based on its block and thread IDs.
     * 2. It iterates over 'numVariables', processing data for each variable.
     * 3. It computes 'outputOffset' to determine the starting index in the 'outputData' array.
     * 4. For each variable, it iterates over 'numSamples', processing data in chunks of 'winSize'.
     * 5. Within each chunk, it calculates a range, 'startIdx' and 'endIdx', to select a subarray
     *    of data centered at the current position 'i'.
     * 6. It creates a temporary array 'windowData' to store the selected subarray.
     * 7. The FFT is applied to 'windowData'.
     * 8. The absolute values of FFT components are calculated.
     * 9. The FFT results for the specified frequencies ('freqs') are stored in 'outputData'.
     * 10. Memory cleanup (in the form of freeing dynamically allocated memory) is suggested but commented out.
     *
     * @note It is assumed that 'winSize' is defined and has a suitable value before calling this kernel.
     */
    __global__ void compute_windowed_fft(float *inputData, float *outputData, int numSamples, int numVariables, int freq_m, int *freqs, int numFreqs) {
        // This kernel computes the windowed FFT of 'inputData' and stores it in 'outputData'.
        // Step 1: Select a window for each thread
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int var = blockIdx.y * blockDim.y + threadIdx.y;
        int stride_x = blockDim.x * gridDim.x * winSize;
        int stride_y = blockDim.y * gridDim.y;
        
        // Step 2: Iterate if the number of variables exceeds the grid size
        for (int v = var; v < numVariables; v += stride_y) {
            // Step 3: starting index in the output array
            int outputOffset = static_cast<int>(ceil((float)numSamples / (float)winSize) * v * numFreqs);
            
            // Step 4: Iterate if the number of samples exceeds the grid size
            for (int i = v * numSamples + idx * winSize; i < (v + 1) * numSamples; i += stride_x) {
                // Step 5: Determine the indexes to select a specific window
                int range = winSize/2;
                int startIdx = (i - range > v * numSamples) ? i - range : v * numSamples;
                int endIdx =  (i + range < (v + 1) * numSamples - 1) ? i + range : (v + 1) * numSamples;

                // Step 6: Window data creation
                Complex windowData[winSize];
                select_subarray(inputData, windowData, startIdx, endIdx, i, winSize);
                // Step 7: windowed FFT computation
                fft(windowData, winSize);

                // Step 8: Absolute value of the FFT
                for (int j = 0; j < winSize / 2 + 1; j++) {
                    windowData[j].x = sqrt(windowData[j].x * windowData[j].x + windowData[j].y * windowData[j].y);
                    windowData[j].y = 0.0f;
                }
                
                // Step 9: Selection of the amplitudes of the desired frequencies
                for (int j = 0; j < numFreqs; j++) {
                    int i_freq = freqs[j]*winSize/freq_m;
                    outputData[outputOffset + idx * numFreqs + j] = windowData[i_freq].x;
                }
                
                // Step 10: Memory cleanup
                // delete[] windowData;
            }
        }
    }
}