#pragma once
#include <cstdint>
#include <string>

class ObjDet
{
public:
    virtual void run(const uint8_t *src) = 0;
    virtual void saveresult(void) = 0;
private:  
    virtual void preprocessing(const uint8_t *src) = 0;
    virtual void inference(void) = 0;
    virtual void postprocessing(void) = 0;
};