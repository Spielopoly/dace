// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
// Minimal C++ shared library providing a high-resolution microsecond timer.
// Compiled at runtime by PythonTimerProvider and loaded via ctypes.

#include <chrono>

extern "C" {
    unsigned long long timer_us() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();
    }
}
