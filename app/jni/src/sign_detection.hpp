#pragma once
#include "obj_detection.hpp"
#include "platform_snpe.hpp"
#include <json.hpp>

class SignDet : ObjDet
{
public:
    SignDet(const nlohmann::json config);
    ~SignDet(void);
    void run(const uint8_t *src);
    void saveresult(void);
private:
    void preprocessing(const uint8_t *src);
    void inference(void);
    void postprocessing(void);
    std::unique_ptr<SNPEContext> platform;
    int netW;
    int netH;
    float meanR;
    float meanG;
    float meanB;
    float stdR;
    float stdG;
    float stdB;
};