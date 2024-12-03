# Compiling code
- Use `make`

# Running code
### Parameters
-  nThreads: number of processes
-  nSamples: sample size used for FFT (must be power of 2)
-  freq: frequency of the sine wave
-  rSampling: sample rate

### To run serial version: ./fast_fourier_transform_serial --nThreads 1 --nSamples <sample_size> --freq <frequency> --rSampling <sample_rate>
### To run parallel version: ./fast_fourier_transform_serial --nThreads <number_of_processes> --nSamples <sample_size> --freq <frequency> --rSampling <sample_rate>
### To run MPI version: mpirun -n <number_of_processes> ./fast_fourier_transform_serial --nSamples <sample_size> --freq <frequency> --rSampling <sample_rate>

