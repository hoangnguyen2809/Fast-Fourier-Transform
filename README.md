# Fast Fourier Transform (FFT) Implementation

A C++ implementation of the Fast Fourier Transform (FFT) algorithm, featuring **serial**, **parallel (multithreaded)**, and **MPI-distributed** versions for performance benchmarking.

## Build
- Use `make`

## Usage
Parameter | Description | Requirement
--nThreads | Number of threads/processes to use | ≥ 1 (ignored in MPI mode)
--nSamples | Sample size used for FFT | Must be a power of 2
--freq | Frequency of the sine wave | Float (e.g., 50.0)
--rSampling | Sampling rate | Float (e.g., 1000.0)

1. Serial version:
   `./fast_fourier_transform_serial --nThreads 1 --nSamples <sample_size> --freq <frequency> --rSampling <sample_rate>`
2. Parallel Version (Multithreading):
   `./fast_fourier_transform_serial --nThreads <number_of_processes> --nSamples <sample_size> --freq <frequency> --rSampling <sample_rate>`
3. MPI version:
   `mpirun -n <number_of_processes> ./fast_fourier_transform_serial --nSamples <sample_size> --freq <frequency> --rSampling <sample_rate>`

## Requirements
- make

- MPI library for running the MPI version

- C++ compiler supporting C++11 or later
