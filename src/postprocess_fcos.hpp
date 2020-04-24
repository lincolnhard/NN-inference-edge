#pragma once

#include <vector>
#include <json.hpp>

// Equal to training.json -> model: max_joints
#define MAX_NUM_VERTEX 6

typedef struct
{
    float x;
    float y;
} Coordinate;

typedef struct
{
    int classId; // seems useless
    std::vector<float> classScores;
    float scoreForSort;
    float centerness;
    Coordinate gridcenter;
    Coordinate vertex[MAX_NUM_VERTEX];
    Coordinate vertexCenter;
} KeyPoint;

class PostprocessFCOS
{
public:
    PostprocessFCOS(const nlohmann::json config);
    ~PostprocessFCOS(void);
    std::vector<std::vector<KeyPoint>> run(std::vector<const float *> &featuremaps);
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
    std::vector<float> classScoreTh;
    std::vector<Coordinate> meshgrid;
    void initMeshgrid();
    Coordinate getAvgCenter(KeyPoint kpt);
};