#pragma once
#ifndef __LOGGER_HPP__
#define __LOGGER_HPP__

#include <iostream>
#include <cstdio>
#include <cstdarg>
#include <string>
#include <ctime>

namespace Logger
{

    // Get the current date and time
    static std::string now()
    {
        // Get the current time
        // Put it in following format
        // [YYYY-MM-DD HH:MM:SS]
        std::time_t rawtime = std::time(0);
        struct tm *timeinfo = std::gmtime(&rawtime);
        if (timeinfo)
        {
            char buffer[80];
            std::strftime(buffer, sizeof(buffer), "[%Y-%m-%d %H:%M:%S]", timeinfo);
            return std::string(buffer);
        }
        return "[1970-01-01 00:00:00]";
        // char buffer[80];
        // time(&rawtime);
        // timeinfo = localtime(&rawtime);
        // strftime(buffer, sizeof(buffer), "[%Y-%m-%d %H:%M:%S]", timeinfo);
        // return std::string(buffer);
    }

    static void cout(const char *format, ...)
    {
        std::string str = now() + ": ";
        str += format;
        va_list args;
        va_start(args, format);
        vprintf(str.c_str(), args);
        va_end(args);
    }
    static void cerr(const char *format, ...)
    {
        std::string str = now() + ": ";
        str += format;
        va_list args;
        va_start(args, format);
        vfprintf(stderr, str.c_str(), args);
        va_end(args);
    }
    static void file(FILE *fid, const char *format, ...)
    {
        std::string str = now() + ": ";
        str += format;
        va_list args;
        va_start(args, format);
        vfprintf(fid, str.c_str(), args);
        va_end(args);
    }
    static void warning(const char *format, ...)
    {
        std::string str = now() + ": Warning - ";
        str += format;
        va_list args;
        va_start(args, format);
        vfprintf(stdout, str.c_str(), args);
        va_end(args);
    }
    static void error(const char *format, ...)
    {
        std::string str = now() + ": Error - ";
        str += format;
        va_list args;
        va_start(args, format);
        vfprintf(stderr, str.c_str(), args);
        va_end(args);
    }
};

#endif // __LOGGER_HPP__
