#include "core/utils.h"
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <vector>

#define DEFAULT_SAMPLE_SIZE "16"
#define DEFAULT_FREQUENCY "5"
#define DEFAULT_SAMPLING_RATE "50"

// Type alias for complex numbers
using Complex = std::complex<double>;

// reverse bits of an index
int reverseBits(int x, int numBits) {
    int reversed = 0;
    for (int i = 0; i < numBits; ++i) {
        reversed <<= 1;
        reversed |= (x & 1);
        x >>= 1;
    }
    return reversed;
}

void fft(std::vector<Complex>& X, std::vector<Complex>& Y, int n) {
    timer serial_timer;
    double time_taken = 0.0;
    serial_timer.start();
    
    //*------------------------------------------------------------------------
    int r = log2(n);

    std::vector<Complex> R(n);
    std::vector<Complex> S(n);
    unsigned int processed_points = 0;

    // ------------------- Bit-Reversal Step -------------------
    for (int i = 0; i < n; ++i) {
        int revIndex = reverseBits(i, r);
        Y[revIndex] = X[i];
    }
    
    // ------------------- FFT Stages -------------------
    for (int m = 1; m <= r; ++m) {
        int step = 1 << m; // 2^m
        Complex w_m = exp(Complex(0, -2.0 * M_PI / step)); // twiddle factor (root of unity)

        for (int k = 0; k < n; k += step) {
            Complex w = 1;
            for (int j = 0; j < step / 2; ++j) {
                Complex t = w * Y[k + j + step / 2];
                Complex u = Y[k + j];
                Y[k + j] = u + t;
                Y[k + j + step / 2] = u - t;
                                
                w *= w_m; // update twiddle factor
                processed_points += 2; // 2 points are processed in each operation
            }
        }
    }
    
    //------------------------------------------------------------------------* 
    time_taken = serial_timer.stop();
    
    // print thread_id, processed_points, time_taken
    std::cout << "thread_id, processed_points, time_taken" << std::endl;
    std::cout << "0, " << processed_points << ", " << std::setprecision(TIME_PRECISION) << time_taken << std::endl;
    // Output the results
    std::cout << "FFT Frequency Bins:" << std::endl;
    int i = 0;
    for (const auto& value : Y) {
        std::cout << "FreqBin[" << i++ << "] = " << value << std::endl;
    }

    std::cout << "Time taken (in seconds) : " << std::setprecision(TIME_PRECISION) << time_taken << "\n";
}

int main(int argc, char *argv[]) {
    // Initialize command line arguments
    cxxopts::Options options("Fast Fourier Transform Computation",
                            "Computes the Discrete Fourier Transform (DFT) of a sequence");
    
    options.add_options(
        "custom",
        {
        {"nThreads", "Number of threads",
        cxxopts::value<uint>()->default_value(DEFAULT_NUMBER_OF_THREADS)},
        {"nSamples", "Number of samples (must be a power of 2 for FFT)",         
        cxxopts::value<uint>()->default_value(DEFAULT_FREQUENCY)}, 
        {"freq", "Frequency of the sine wave in Hz",         
        cxxopts::value<double>()->default_value(DEFAULT_FREQUENCY)},     
        {"rSampling", "Sampling rate in Hz (samples per second)",
        cxxopts::value<double>()->default_value(DEFAULT_SAMPLING_RATE)},
        });

    auto cl_options = options.parse(argc, argv);
    uint n_threads = cl_options["nThreads"].as<uint>();
    if (n_threads != 1) {
        std::cout << "Serial version. Number of threads should be equal to 1. Terminating..." << std::endl;
        return 1;
    }
    uint n = cl_options["nSamples"].as<uint>();
    double frequency = cl_options["freq"].as<double>();
    double samplingRate = cl_options["rSampling"].as<double>();

    std::cout << "Sample Size: " << n << std::endl;
    std::cout << "Number of threads: " << n_threads << std::endl;
    std::cout << "Frequency of the sine wave: " << frequency << " Hz" << std::endl;
    std::cout << "Sampling rate: " << samplingRate << " Hz" << std::endl;
    std::cout << "Initializing Sine Wave..." << std::endl;

    std::vector<Complex> X = generateSineWave(n, frequency, samplingRate);
    
    int size = X.size();
    std::vector<Complex> Y(size);
    
    std::cout << "Computing FFT..." << std::endl;
    fft(X, Y, size);
    
    return 0;
}
