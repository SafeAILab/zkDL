#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::duration<double> totalTime;
    bool isRunning;

public:
    Timer() : totalTime(0.0), isRunning(false) {}

    void start() {
        if (!isRunning) {
            startTime = std::chrono::high_resolution_clock::now();
            isRunning = true;
        }
    }

    void stop() {
        if (isRunning) {
            std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
            totalTime += std::chrono::duration_cast<std::chrono::duration<double>>(endTime - startTime);
            isRunning = false;
        }
    }

    void reset() {
        isRunning = false;
        totalTime = std::chrono::duration<double>(0.0);
    }

    double getTotalTime() const {
        return totalTime.count();
    }
};

#endif  // TIMER_HPP_INCLUDED
