#pragma once

#include <vector>
#include <json.hpp>
#include "postprocess.hpp"

#define MAX_NUM_VERTEX 4

typedef struct
{
    int clsId;
    float x0;
    float y0;
    float x1;
    float y1;
    float x2;
    float y2;
    float x3;
    float y3;
    float x4;
    float y4;
    float x5;
    float y5;
    float score;
} ScoreVertices; // maybe its inappropriate to put this definition here

typedef struct
{
    float x;
    float y;
} Coordinate;

typedef struct
{
    int classId;
    float classScore;
    float score;
    Coordinate center;
    Coordinate vertex[MAX_NUM_VERTEX];
    Coordinate vertexCenter;
} KeyPoint;

class PostprocessFCOS : Postprocess
{
public:
    PostprocessFCOS(const nlohmann::json config);
    ~PostprocessFCOS(void);
    void setInput(std::vector<float *> featuremaps);
    void run(std::vector<std::vector<ScoreVertices>> &predResult);
private:
    int netW;
    int netH;
    int featW;
    int featH;
    int numClass;
    int stride;
    float scale;
    int topk;
    float nmsTh;
    float classScoreTh;
    float *classScoreMap;
    float *centernessMap;
    float *regressionMap;
    std::vector<Coordinate> meshgrid;
    void initMeshgrid();
    Coordinate getAvgCenter(KeyPoint kpt);
};