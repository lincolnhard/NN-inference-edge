#include <string>
#include <memory>
#include <atomic>
#include <iomanip>
#include "log_stream.hpp"

std::atomic<uint64_t> threadCount(1);
thread_local uint64_t _threadId = 0;

static uint64_t threadId()
{
    if (_threadId < 1)
    {
        _threadId = threadCount.fetch_add(1, std::memory_order_release);
    }
    return _threadId;
}

LogStream::LogStream(std::shared_ptr<spdlog::logger> log, spdlog::level::level_enum level) : m_Logger(log), m_Level(level), m_Stream()
{
    m_Stream<<std::setprecision(12) << '[' << threadId() << "] ";
}

LogStream::~LogStream()
{
    std::string s = m_Stream.str();
    if (!s.empty())
    {
        m_Logger->log(m_Level, m_Stream.str());
    }
}

void LogStream::Flush()
{
    m_Logger->log(m_Level, m_Stream.str());
    m_Stream.str("");
    m_Stream.clear();
}

ExceptionStream::ExceptionStream() : m_Stream()
{

}

ExceptionStream::~ExceptionStream()
{
    std::string s = m_Stream.str();
    if (!s.empty())
    {
        Flush();
    }
}

std::shared_ptr<LogStream> LogStream::logWithLevel(std::shared_ptr<spdlog::logger> log, spdlog::level::level_enum level)
{
    return std::make_shared<LogStream>(log, level);
}

std::shared_ptr<LogStream> LogStream::logInfo(std::shared_ptr<spdlog::logger> log)
{
    return LogStream::logWithLevel(log, spdlog::level::info);
}

std::shared_ptr<LogStream> LogStream::logDebug(std::shared_ptr<spdlog::logger> log)
{
    return LogStream::logWithLevel(log, spdlog::level::debug);
}

std::shared_ptr<LogStream> LogStream::logError(std::shared_ptr<spdlog::logger> log)
{
    return LogStream::logWithLevel(log, spdlog::level::err);
}

std::shared_ptr<LogStream> LogStream::logWarn(std::shared_ptr<spdlog::logger> log)
{
    return LogStream::logWithLevel(log, spdlog::level::warn);
}

void LogStream::logException(std::exception_ptr eptr, std::shared_ptr<spdlog::logger> log)
{
    if (eptr)
    {
        std::rethrow_exception(eptr);
    }
}

std::shared_ptr<ExceptionStream> LogStream::throwExeption()
{
    return std::make_shared<ExceptionStream>();
}

void ExceptionStream::Flush()
{
    throw std::runtime_error(m_Stream.str());
}
