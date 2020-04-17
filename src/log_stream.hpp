#pragma once
#include <spdlog/spdlog.h>
#include <sstream>


#define SLOG_INFO LogStream::logInfo(LOG)
#define SLOG_WARN LogStream::logWarn(LOG)
#define SLOG_ERROR LogStream::logError(LOG)
#define SLOG_DEBUG LogStream::logDebug(LOG)
#define THROW_EXCEPTION LogStream::throwExeption()<<__func__<<'('<<__FILE__<<"): "

class ExceptionStream
{
private:
    std::stringstream m_Stream;
public:
    ExceptionStream();
    ~ExceptionStream();
    void Flush();
    template<typename T> 
    void Append(const T& str)
    {
        m_Stream << str;
    }
    
};

class LogStream
{
private:
    std::shared_ptr<spdlog::logger> m_Logger;
    spdlog::level::level_enum m_Level;
    std::stringstream m_Stream;
public:
    LogStream(std::shared_ptr<spdlog::logger> log, spdlog::level::level_enum level);
    ~LogStream();
    void Flush();
    template<typename T> 
    void Append(const T& str)
    {
        m_Stream << str;
    }

    static void logException(std::exception_ptr eptr,std::shared_ptr<spdlog::logger> log);
    static void logException(std::exception_ptr eptr);
    static std::shared_ptr<LogStream> logWithLevel(std::shared_ptr<spdlog::logger> log, spdlog::level::level_enum level);
    static std::shared_ptr<LogStream> logInfo(std::shared_ptr<spdlog::logger> log);
    static std::shared_ptr<LogStream> logError(std::shared_ptr<spdlog::logger> log);
    static std::shared_ptr<LogStream> logWarn(std::shared_ptr<spdlog::logger> log);
    static std::shared_ptr<LogStream> logDebug(std::shared_ptr<spdlog::logger> log);
    static std::shared_ptr<ExceptionStream> throwExeption();
};

inline LogStream& operator<<(LogStream& os,std::ostream& (*fun)(std::ostream&))
{
    os.Flush();    
    return os; 
}

template<typename T> 
inline LogStream& operator<<(LogStream& os, const T& dt)
{
    os.Append(dt);    
    return os;  
} 

inline std::shared_ptr<LogStream> operator<<(std::shared_ptr<LogStream> os,std::ostream& (*fun)(std::ostream&))
{
    os->Flush();    
    return os; 
}

template<typename T>
inline std::shared_ptr<LogStream> operator<<(std::shared_ptr<LogStream> os, const T& dt)
{
    os->Append(dt);
    return os;
}

inline std::shared_ptr<LogStream> operator<<(std::shared_ptr<LogStream> os, const std::chrono::time_point<std::chrono::steady_clock,std::chrono::nanoseconds> dt)
{
    os->Append(std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(dt.time_since_epoch()).count());
    os->Append("ms");
    return os;
}

inline std::shared_ptr<LogStream> operator<<(std::shared_ptr<LogStream> os, const std::chrono::nanoseconds dt)
{
    os->Append(std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(dt).count());
    os->Append("ms");
    return os;
}