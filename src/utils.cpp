#include "utils.hpp"

#include <fstream>
#include <sys/resource.h>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "log_stream.hpp"

static auto LOG = spdlog::stdout_color_mt("UTILS");

namespace gallopwave
{

uint32_t SysRes::getPeakRSS(void)
{
    struct rusage rusg;
    getrusage(RUSAGE_SELF, &rusg);
	return static_cast<uint32_t>(rusg.ru_maxrss * 1024);
}

uint32_t SysRes::getCurrentRSS(void)
{
    std::ifstream fin;
    fin.open("/proc/self/statm");
    uint32_t prgramsize = 0;
    uint32_t residentsize = 0;
    uint32_t sharedsize = 0;
    uint32_t textsize = 0;
    uint32_t libsize = 0;
    uint32_t stacksize = 0;
    uint32_t dtsize = 0;
    fin >> prgramsize >> residentsize >> sharedsize >>
    textsize >> libsize >> stacksize >> dtsize;
	fin.close();
	return static_cast<uint32_t>(residentsize * sysconf( _SC_PAGESIZE));
}

void SysRes::markCpuState(void)
{
    std::ifstream fin;
    fin.open("/proc/stat");
    std::string dev;
    uint32_t userJiffies = 0;
    uint32_t niceJiffies = 0;
    uint32_t systemJiffies = 0;
    uint32_t idleJiffies = 0;
    uint32_t iowaitJiffies = 0;
    uint32_t irqJiffies = 0;
    uint32_t softirqJiffies = 0;

    fin >> dev >> userJiffies >> niceJiffies >> systemJiffies >>
    idleJiffies >> iowaitJiffies >> irqJiffies >> softirqJiffies;

    workJiffies = userJiffies + niceJiffies + systemJiffies;
    totalJiffies = workJiffies + idleJiffies + iowaitJiffies + irqJiffies + softirqJiffies;
}

double SysRes::getCpuUsage(void)
{
    std::ifstream fin;
    fin.open("/proc/stat");
    std::string dev;
    uint32_t userJiffies;
    uint32_t niceJiffies;
    uint32_t systemJiffies;
    uint32_t idleJiffies;
    uint32_t iowaitJiffies;
    uint32_t irqJiffies;
    uint32_t softirqJiffies;

    fin >> dev >> userJiffies >> niceJiffies >> systemJiffies >>
    idleJiffies >> iowaitJiffies >> irqJiffies >> softirqJiffies;


    uint32_t workOverPeriod = userJiffies + niceJiffies + systemJiffies - workJiffies;
    uint32_t totalOverPeriod = userJiffies + niceJiffies + systemJiffies + idleJiffies +
                            iowaitJiffies + irqJiffies + softirqJiffies - totalJiffies;
    double cpusage = 1.0 * workOverPeriod / totalOverPeriod;
    return cpusage;
}

}