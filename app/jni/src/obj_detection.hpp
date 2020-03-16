#pragma once
#include <cstdint>
#include <string>

class ObjDet
{
public:
    virtual void run(uint8_t *src, const int srcw, const int srch, std::string dstpath) = 0;
    virtual void saveresult(uint8_t *src, const int srcw, const int srch, std::string dstpath) = 0;
private:  
    virtual void preprocessing(const uint8_t *src) = 0;
    virtual void inference(void) = 0;
    virtual void postprocessing(void) = 0;
};