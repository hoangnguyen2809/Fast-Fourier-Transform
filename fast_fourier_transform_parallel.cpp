#include "core/utils.h"
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <thread>
#include <fstream>

#define DEFAULT_SAMPLE_SIZE "16"
#define DEFAULT_FREQUENCY "5.0"
#define DEFAULT_SAMPLING_RATE "50.0"

// Type alias for complex numbers
using Complex = std::complex<double>;

// Constants


// Global variables for input and output
std::vector<Complex> X_global;   // Input vector
std::vector<Complex> Y_global;   // Output vector after FFT
int n_global;                    // Size of the input (must be a power of 2)
int r_global;                    // Number of bits for indexing

// Function to reverse the bits of an index
int reverseBits(int x, int numBits) {
    int reversed = 0;
    for (int i = 0; i < numBits; ++i) {
        reversed <<= 1;
        reversed |= (x & 1);
        x >>= 1;
    }
    return reversed;
}

// Structure to hold data for each thread
struct ThreadData {
    int rank;                        // Thread identifier (0 to NUM_THREADS-1)
    int num_threads;                 // Total number of threads
    std::vector<Complex>* X;         // Pointer to input vector
    std::vector<Complex>* Y;         // Pointer to output vector
    int n;                           // Size of the input
    int r;                           // Number of bits
    double time_taken;               // Time taken by the thread
    int processed_points;            // Number of points processed by the thread
    CustomBarrier* barrier;          // Pointer to the barrier
};

// Thread function to perform parallel FFT
void thread_func(ThreadData* data) {
    // Start timing for the thread
    timer thread_timer;
    thread_timer.start();

    // ------------------- Bit-Reversal Step -------------------
    for (int i = data->rank; i < data->n; i += data->num_threads) {
        int revIndex = reverseBits(i, data->r);
        (*data->Y)[revIndex] = (*data->X)[i];
        data->processed_points += 1;
    }

    // Synchronize all threads after bit-reversal
    data->barrier->wait();

    // ------------------- FFT Stages -------------------
    for (int m = 1; m <= data->r; ++m) {
        int step = 1 << m; // 2^m
        Complex w_m = std::exp(Complex(0, -2.0 * M_PI / step)); // Twiddle factor

        for (int k = 0; k < data->n; k += step) {
            // Assign k indices to threads using modulo operation
            if (((k / step) % data->num_threads) != data->rank) continue;

            Complex w = 1;
            for (int j = 0; j < step / 2; ++j) {
                Complex t = w * (*data->Y)[k + j + step / 2];
                Complex u = (*data->Y)[k + j];
                (*data->Y)[k + j] = u + t;
                (*data->Y)[k + j + step / 2] = u - t;
                w *= w_m;
                data->processed_points += 2; // Two operations per butterfly
            }
        }

        // Synchronize all threads after each FFT stage
        data->barrier->wait();
    }

    // End timing for the thread
    data->time_taken = thread_timer.stop();
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
    // ------------------- Initialization -------------------
    // Sample input (size must be a power of 2)
    
    
    auto cl_options = options.parse(argc, argv);
    uint n_threads = cl_options["nThreads"].as<uint>();
    if (n_threads == 0) {
        std::cout << "Number of threads cannot be 0. Terminating..." << std::endl;
        return 1;
    }
    uint n = cl_options["nSamples"].as<uint>();
    double frequency = cl_options["freq"].as<double>();
    double samplingRate = cl_options["rSampling"].as<double>();

    std::vector<Complex> X = generateSineWave(n, frequency, samplingRate);
    
    n_global = X.size();
    r_global = log2(n_global);
    // Check if n is a power of 2
    if ((1 << r_global) != n_global) {
        std::cerr << "Error: Input size must be a power of 2." << std::endl;
        return -1;
    }
    std::cout << "Sample Size: " << n << std::endl;
    std::cout << "Number of threads: " << n_threads << std::endl;
    std::cout << "Frequency of the sine wave: " << frequency << " Hz" << std::endl;
    std::cout << "Sampling rate: " << samplingRate << " Hz" << std::endl;
    std::cout << "Initializing Sine Wave..." << std::endl;

    // Initialize global input and output vectors
    X_global = X;
    Y_global.resize(n_global);

    // Initialize the barrier for thread synchronization
    CustomBarrier barrier(n_threads);

    // ------------------- Thread Creation -------------------
    std::thread threads[n_threads];
    ThreadData thread_data[n_threads];

    // Create threads and assign work
    for (int t = 0; t < n_threads; ++t) {
        thread_data[t].rank = t;
        thread_data[t].num_threads = n_threads;
        thread_data[t].X = &X_global;
        thread_data[t].Y = &Y_global;
        thread_data[t].n = n_global;
        thread_data[t].r = r_global;
        thread_data[t].time_taken = 0.0;
        thread_data[t].processed_points = 0;
        thread_data[t].barrier = &barrier;

        threads[t] = std::thread(thread_func, &thread_data[t]);
    }

    // ------------------- Thread Joining -------------------
    // Wait for all threads to complete
    for (int t = 0; t < n_threads; ++t) {
        threads[t].join();
    }

    // ------------------- Timing and Output -------------------
    // Compute the total time taken (maximum time among all threads)
    double total_time = 0.0;
    for (int t = 0; t < n_threads; ++t) {
        if (thread_data[t].time_taken > total_time) {
            total_time = thread_data[t].time_taken;
        }
    }
    
    // Output the results
    std::cout << "Computing FFT..." << std::endl;
    std::cout << "thread_id, processed_points, time_taken" << std::endl;
    for (int t = 0; t < n_threads; ++t) {
        std::cout << t << ", " << thread_data[t].processed_points << ", ";
        std::cout << std::fixed << std::setprecision(TIME_PRECISION) << thread_data[t].time_taken << std::endl;
    }

    std::cout << "FFT Frequency Bins:" << std::endl;
    int position = 0;
    int step = n/6;
    for (int x = 0; x < 6; x++) {
        std::cout << "FreqBin[" << position << "] = " << Y_global[position] << std::endl;
        position += step;
    }
    // for (const auto& value : Y_global) {
    //     std::cout << std::fixed << std::setprecision(5) << "(" << value.real() << "," << value.imag() << ")" << std::endl;
    // }

    std::cout << "Time taken (in seconds) : " << std::fixed << std::setprecision(TIME_PRECISION) << total_time << "\n";
    std::ofstream outputFile("fft_parallel_output.csv");
    outputFile << "Frequency Bin,Magnitude,Phase\n"; // CSV header

    for (size_t i = 0; i < Y_global.size(); ++i) {
        double magnitude = std::abs(Y_global[i]);
        double phase = std::arg(Y_global[i]);
        outputFile << i << "," << magnitude << "," << phase << "\n";
    }

    outputFile.close();
    std::cout << "FFT output saved to fft_parallel_output.csv\n";
    std::cout << "Run python3 plotting.py <filename> to plot the output\n";
    return 0;

    return 0;
}