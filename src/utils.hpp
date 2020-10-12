#pragma once

#include <cstdint>

namespace gallopwave
{

class SysRes
{
public:
    uint32_t getPeakRSS(void);
    uint32_t getCurrentRSS(void);
    void markCpuState(void);
    double getCpuUsage(void);
private:
    uint32_t workJiffies;
    uint32_t totalJiffies;
};

}