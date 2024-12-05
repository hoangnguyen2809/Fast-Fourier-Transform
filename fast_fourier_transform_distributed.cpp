#include "core/utils.h"
#include <mpi.h>
#include <iomanip>
#include <iostream>
#include <vector>
#include <complex>
#include <fstream>

#define DEFAULT_SAMPLE_SIZE "16"
#define DEFAULT_FREQUENCY "5.0"
#define DEFAULT_SAMPLING_RATE "50.0"
#define DEFAULT_NUMBER_OF_THREADS "1"

// Type alias for complex numbers
using Complex = std::complex<double>;

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

// Distributed FFT implementation
void distributedFFT(std::vector<Complex>& X, std::vector<Complex>& Y, int n, int rank, int num_procs, MPI_Comm comm) {
    timer distributed_timer;
    double local_time_taken = 0.0;
    int local_processed_points = 0;

    // Start timing
    distributed_timer.start();

    // Prepare work distribution variables
    int r = log2(n);
    std::vector<Complex> local_Y(n, Complex(0, 0));
    std::vector<Complex> global_Y(n, Complex(0, 0));

    // ------------------- Bit-Reversal Step -------------------
    for (int i = rank; i < n; i += num_procs) {
        int revIndex = reverseBits(i, r);
        local_Y[i] = X[revIndex];
    }

    // Reduce to get complete global bit-reversed array
    MPI_Allreduce(local_Y.data(), global_Y.data(), 2 * n, MPI_DOUBLE, MPI_SUM, comm);
    Y = global_Y;

    // ------------------- FFT Stages -------------------
    for (int m = 1; m <= r; ++m) {
        int step = 1 << m; // 2^m
        Complex w_m = std::exp(Complex(0, -2.0 * M_PI / step)); // Twiddle factor
        std::fill(local_Y.begin(), local_Y.end(), Complex(0, 0));

        for (int k = 0; k < n; k += step) {
            if (((k / step) % num_procs) != rank) continue;

            Complex w = 1;
            for (int j = 0; j < step / 2; ++j) {
                Complex t = w * Y[k + j + step / 2];
                Complex u = Y[k + j];
                local_Y[k + j] = u + t;
                local_Y[k + j + step / 2] = u - t;
                w *= w_m;
                local_processed_points += 2;
            }
        }

        MPI_Allreduce(local_Y.data(), Y.data(), 2 * n, MPI_DOUBLE, MPI_SUM, comm);
    }

    local_time_taken = distributed_timer.stop();

    // Gather timing and processed points to rank 0
    std::vector<int> processed_points(num_procs);
    std::vector<double> time_taken(num_procs);

    MPI_Gather(&local_processed_points, 1, MPI_INT, processed_points.data(), 1, MPI_INT, 0, comm);
    MPI_Gather(&local_time_taken, 1, MPI_DOUBLE, time_taken.data(), 1, MPI_DOUBLE, 0, comm);

    if (rank == 0) {
        // Print thread statistics
        std::cout << "thread_id, processed_points, time_taken\n";
        for (int i = 0; i < num_procs; ++i) {
            std::cout << i << ", " << processed_points[i] << ", "
                      << std::fixed << std::setprecision(TIME_PRECISION) << time_taken[i] << "\n";
        }
        std::cout << "FFT Frequency Bins:\n";
        int position = 0;
        int step = n / 6;
        for (int x = 0; x < 6; x++) {
            std::cout << "FreqBin[" << position << "] = " << Y[position] << "\n";
            position += step;
        }
    }
}

int main(int argc, char *argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get process rank and total number of processes
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    uint n;
    double frequency, samplingRate;

    // Only rank 0 parses command line arguments
    if (rank == 0) {
            // Initialize command line arguments
            cxxopts::Options options("Distributed Fast Fourier Transform Computation",
                                    "Computes the Discrete Fourier Transform (DFT) of a sequence");
            
            options.add_options(
                "custom",
                {
                {"nSamples", "Number of samples (must be a power of 2 for FFT)",         
                cxxopts::value<uint>()->default_value(DEFAULT_SAMPLE_SIZE)}, 
                {"freq", "Frequency of the sine wave in Hz",         
                cxxopts::value<double>()->default_value(DEFAULT_FREQUENCY)},     
                {"rSampling", "Sampling rate in Hz (samples per second)",
                cxxopts::value<double>()->default_value(DEFAULT_SAMPLING_RATE)},
                });

            auto cl_options = options.parse(argc, argv);
            
            // Extract parameters
            n = cl_options["nSamples"].as<uint>();
            frequency = cl_options["freq"].as<double>();
            samplingRate = cl_options["rSampling"].as<double>();

            std::cout << "Sample Size: " << n << std::endl;
            std::cout << "Number of processes: " << num_procs << std::endl;
            std::cout << "Frequency of the sine wave: " << frequency << " Hz" << std::endl;
            std::cout << "Sampling rate: " << samplingRate << " Hz" << std::endl;
            std::cout << "Initializing Sine Wave..." << std::endl;
    }

    // Broadcast parameters to all processes
    MPI_Bcast(&n, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&frequency, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&samplingRate, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Verify input size is a power of 2
    int r = log2(n);
    if ((1 << r) != n) {
        if (rank == 0) {
            std::cerr << "Error: Input size must be a power of 2." << std::endl;
        }
        MPI_Finalize();
        return -1;
    }

    // Generate sine wave on rank 0 and distribute to all processes
    std::vector<Complex> X(n);
    std::vector<Complex> Y(n);

    if (rank == 0) {
        X = generateSineWave(n, frequency, samplingRate);
    }

    // Broadcast input sine wave to all processes
    MPI_Bcast(X.data(), 2 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
        std::cout << "Computing FFT... " << std::endl;

    // Start total timing
    timer total_timer;
    if (rank == 0) {
        total_timer.start();
    }

    // Perform distributed FFT
    distributedFFT(X, Y, n, rank, num_procs, MPI_COMM_WORLD);

    // Stop total timing
    double total_time_taken = 0.0;
    if (rank == 0) {
        total_time_taken = total_timer.stop();
        std::cout << "Total time taken (in seconds): "
                  << std::fixed << std::setprecision(TIME_PRECISION) << total_time_taken << "\n";
    }

    // Output to CSV from rank 0
    // if (rank == 0) {
    //     std::ofstream outputFile("fft_distributed_output.csv");
    //     outputFile << "Frequency Bin,Magnitude,Phase\n"; // CSV header

    //     for (size_t i = 0; i < Y.size(); ++i) {
    //         double magnitude = std::abs(Y[i]);
    //         double phase = std::arg(Y[i]);
    //         outputFile << i << "," << magnitude << "," << phase << "\n";
    //     }

    //     outputFile.close();
    //     std::cout << "FFT output saved to fft_distributed_output.csv\n";
    //     std::cout << "Run python3 plotting.py <filename> to plot the output\n";
    // }

    MPI_Finalize();
    return 0;
}