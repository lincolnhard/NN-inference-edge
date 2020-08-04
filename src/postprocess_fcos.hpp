#pragma once

#include <vector>
#include <json.hpp>


typedef struct
{
    float x;
    float y;
} Coordinate;

// typedef struct
// {
//     int classId;
//     std::vector<float> classScores;
//     float scoreForSort;
//     float centerness;
//     Coordinate gridcenter;
//     Coordinate vertex[MAX_NUM_VERTEX];
//     Coordinate vertexCenter;
// } KeyPoint;

typedef struct
{
    int classId;
    std::vector<float> classScores;
    float scoreForSort;
    float centerness;
    Coordinate gridcenter;
    Coordinate vertexTL;
    Coordinate vertexBR;
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
    std::vector<Coordinate> meshgrid; // mlvl_points
    void initMeshgrid();
    Coordinate getAvgCenter(KeyPoint kpt);
};