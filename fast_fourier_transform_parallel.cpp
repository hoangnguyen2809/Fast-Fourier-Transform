#include "core/utils.h"
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <thread>
#include <iomanip>

// Type alias for complex numbers
using Complex = std::complex<double>;

// Constants
#define NUM_THREADS 4

// Global variables for input and output
std::vector<Complex> X_global;   // Input vector
std::vector<Complex> Y_global;   // Output vector after FFT
int n_global;                    // Size of the input (must be a power of 2)
int r_global;                    // Number of bits for indexing

// Barrier for thread synchronization
CustomBarrier barrier(NUM_THREADS);

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
        if (revIndex != i) {
            (*data->Y)[revIndex] = (*data->X)[i];
            data->processed_points += 1;
        } else {
            // Handle the case when revIndex == i
            (*data->Y)[i] = (*data->X)[i];
            data->processed_points += 1;
        }
    }

    // Synchronize all threads after bit-reversal
    data->barrier->wait();

    // ------------------- FFT Stages -------------------
    for (int m = 1; m <= data->r; ++m) {
        int step = 1 << m; // 2^m
        Complex w_m = std::exp(Complex(0, -2.0 * M_PI / step)); // Twiddle factor

        // Each thread processes a subset of k's
        for (int k = data->rank * (data->n / data->num_threads); k < data->n; k += data->num_threads * (data->n / data->num_threads)) {
            for (int j = 0; j < step / 2; ++j) {
                if (k + j + step / 2 < data->n) {
                    Complex w = std::pow(w_m, j); // Compute twiddle factor for this j
                    Complex t = w * (*data->Y)[k + j + step / 2];
                    Complex u = (*data->Y)[k + j];
                    (*data->Y)[k + j] = u + t;
                    (*data->Y)[k + j + step / 2] = u - t;
                    data->processed_points += 2; // Two operations per butterfly
                }
            }
        }

        // Synchronize all threads after each FFT stage
        data->barrier->wait();
    }

    // End timing for the thread
    data->time_taken = thread_timer.stop();
}

int main() {
    // ------------------- Initialization -------------------
    // Sample input (size must be a power of 2)
    std::vector<Complex> X = { 1, 2, 3, 4 };
    n_global = X.size();
    r_global = log2(n_global);

    // Check if n is a power of 2
    if ((1 << r_global) != n_global) {
        std::cerr << "Error: Input size must be a power of 2." << std::endl;
        return -1;
    }

    // Initialize global input and output vectors
    X_global = X;
    Y_global.resize(n_global);

    // ------------------- Thread Creation -------------------
    std::thread threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    // Initialize the barrier for thread synchronization
    CustomBarrier barrier(NUM_THREADS);

    // Create threads and assign work
    for (int t = 0; t < NUM_THREADS; ++t) {
        thread_data[t].rank = t;
        thread_data[t].num_threads = NUM_THREADS;
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
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads[t].join();
    }

    // ------------------- Timing and Output -------------------
    // Compute the total time taken (maximum time among all threads)
    double total_time = 0.0;
    for (int t = 0; t < NUM_THREADS; ++t) {
        if (thread_data[t].time_taken > total_time) {
            total_time = thread_data[t].time_taken;
        }
    }

    // Output the results
    std::cout << "Number of processes : " << NUM_THREADS << "\n";
    std::cout << "rank, processed_points, time_taken" << std::endl;
    for (int t = 0; t < NUM_THREADS; ++t) {
        std::cout << t << ", " << thread_data[t].processed_points << ", ";
        std::cout << std::fixed << std::setprecision(TIME_PRECISION) << thread_data[t].time_taken << std::endl;
    }

    std::cout << "FFT result:" << std::endl;
    for (const auto& value : Y_global) {
        std::cout << value << std::endl;
    }

    std::cout << "Time taken (in seconds) : " << std::fixed << std::setprecision(TIME_PRECISION) << total_time << "\n";

    return 0;
}