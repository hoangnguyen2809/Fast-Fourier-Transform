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

void fft(std::vector<Complex>& X) {
    int n = X.size();
    int r = log2(n);

    std::vector<Complex> Y(n);
    for (int i = 0; i < n; ++i) {
        int revIndex = reverseBits(i, r);
        Y[revIndex] = X[i];
    }

    for (int m = 1; m <= r; ++m) {
        int step = 1 << m; // 2^m
        Complex w_m = exp(Complex(0, -2.0 * M_PI / step)); 

        for (int k = 0; k < n; k += step) {
            Complex w = 1; 
            for (int j = 0; j < step / 2; ++j) {
                Complex t = w * Y[k + j + step / 2];
                Complex u = Y[k + j];
                Y[k + j] = u + t;
                Y[k + j + step / 2] = u - t;
                
                w *= w_m;
            }
        }
    }

    // Copy the result back to X
    for (int i = 0; i < n; ++i) {
        X[i] = Y[i];
    }
}

int main() {
    // Sample input (size must be a power of 2)
    std::vector<Complex> X = { 1, 2, 3, 4 };

    // Perform FFT
    fft(X);

    // Output the results
    std::cout << "FFT result:" << std::endl;
    for (const auto& value : X) {
        std::cout << value << std::endl;
    }

    return 0;
}
