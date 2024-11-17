#include "core/utils.h"
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <complex>
#include <cmath>

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

    for (int i = 0; i < n; ++i) {
        int revIndex = reverseBits(i, r);
        Y[revIndex] = X[i];
    }
    
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
            }
        }
    }

    
    //------------------------------------------------------------------------* 
    time_taken = serial_timer.stop();
    
    // Output the results
    std::cout << "FFT result:" << std::endl;
    for (const auto& value : Y) {
        std::cout << value << std::endl;
    }
    
    std::cout << "Time taken (in seconds) : " << std::setprecision(TIME_PRECISION) << time_taken << "\n";
}

int main() {
    // Initialize command line arguments
    cxxopts::Options options("Fast Fourier Transform Computation",
                            "Computes the Discrete Fourier Transform (DFT) of a sequence");
    

    // Sample input (size must be a power of 2)
    std::vector<Complex> X = { 1, 2, 3, 4 };
    int n = X.size();
    std::vector<Complex> Y(n);
    
    fft(X, Y, n);
    
    return 0;
}
