#ifndef UTILS_H
#define UTILS_H

#include "cxxopts.h"
#include "get_time.h"
#include <iostream>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <limits.h>
#include <cmath>
#include <complex>

#define intV int32_t
#define uintV int32_t
#define UINTV_MAX INT_MAX

#define intE int32_t
#define uintE int32_t

#define DEFAULT_NUMBER_OF_THREADS "1"
#define DEFAULT_MAX_ITER "20"
#define TIME_PRECISION 5
#define VAL_PRECISION 14
#define THREAD_LOGS 0
#define DEFAULT_STRATEGY "1"
#define DEFAULT_GRANULARITY "1"
#define ADDITIONAL_TIMER_LOGS 1 


using Complex = std::complex<double>;

struct CustomBarrier
{
    int num_of_workers_;
    int current_waiting_;
    int barrier_call_;
    std::mutex my_mutex_;
    std::condition_variable my_cv_;

    CustomBarrier(int t_num_of_workers) : num_of_workers_(t_num_of_workers), current_waiting_(0), barrier_call_(0) {}

    void wait()
    {
        std::unique_lock<std::mutex> u_lock(my_mutex_);
        int c = barrier_call_;
        current_waiting_++;
        if (current_waiting_ == num_of_workers_)
        {
            current_waiting_ = 0;
            // unlock and send signal to wake up
            barrier_call_++;
            u_lock.unlock();
            my_cv_.notify_all();
            return;
        }
        my_cv_.wait(u_lock, [&]{return (c != barrier_call_); });
        //  Condition has been reached. return
    }
};

std::vector<Complex> generateSineWave(size_t n, double frequency, double samplingRate) {
    std::vector<Complex>signal(n);

    for (size_t i = 0; i < n; ++i) {
        // Calculate time at sample i
        double t = static_cast<double>(i) / samplingRate;

        // Generate the real part as a sine wave, imaginary part is zero
        double value = sin(2 * M_PI * frequency * t);
        signal[i] = Complex(value, 0.0);
    }

    return signal;
}

#endif
