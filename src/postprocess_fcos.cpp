#include <assert.h>
#include "postprocess_fcos.hpp"
#include <deque>

PostprocessFCOS::~PostprocessFCOS()
{

}

PostprocessFCOS::PostprocessFCOS(const nlohmann::json config)
{
    netW = config["net_width"].get<int>();
    netH = config["net_height"].get<int>();
    featW = config["feature_width"].get<int>();
    featH = config["feature_height"].get<int>();
    stride = config["stride"].get<int>();
    topk = config["topK"].get<int>();
    numClass = config["num_class"].get<int>();
    scale = config["scale"].get<float>();
    nmsTh = config["nms_threshold"].get<float>();
    classScoreTh = config["class_score_threshold"].get<std::vector<float> >();

    assert(netW == featW * stride);
    assert(netH == featH * stride);
    assert(classScoreTh.size() == numClass);

    initMeshgrid();
}

void PostprocessFCOS::initMeshgrid()
{
    const int FEAT_PLANE_SIZE = featW * featH;
    int widx = 0;
    int hidx = 0;
    for (int i = 0; i < FEAT_PLANE_SIZE; ++i)
    {
        Coordinate coord;
        coord.x = float(widx * stride + (int)(stride * 0.5));
        coord.y = float(hidx * stride + (int)(stride * 0.5));
        meshgrid.push_back(coord);

        ++widx;
        if (widx >= featW)
        {
            widx = 0;
            ++hidx;
        }
    }
}

Coordinate PostprocessFCOS::getAvgCenter(KeyPoint kpt)
{
    Coordinate avgpt;
    avgpt.x = 0.25f * (kpt.vertex[0].x + kpt.vertex[1].x + kpt.vertex[2].x + kpt.vertex[3].x);
    avgpt.y = 0.25f * (kpt.vertex[0].y + kpt.vertex[1].y + kpt.vertex[2].y + kpt.vertex[3].y);
    return avgpt;
}

std::vector<std::vector<KeyPoint>> PostprocessFCOS::run(std::vector<const float *> &featuremaps)
{
    const float *classScoreMap = featuremaps[0];
    const float *centernessMap = featuremaps[1];
    const float *regressionMap = featuremaps[2];
    const float *occlusionsMap = featuremaps[3];

    // score argmax
    std::vector<KeyPoint> keypoints;
    const int FEAT_PLANE_SIZE = featW * featH;
    for (int posIdx = 0; posIdx < FEAT_PLANE_SIZE; ++posIdx)
    {
        int startclsIdx = numClass * posIdx;
        float bestClassScore = classScoreMap[startclsIdx];
        int bestClassId = 0;
        for (int clsIdx = 1; clsIdx < numClass; ++clsIdx)
        {
            float currentClassScore = classScoreMap[startclsIdx + clsIdx];
            if (currentClassScore > bestClassScore)
            {
                bestClassScore = currentClassScore;
                bestClassId = clsIdx;
            }
        }

        int vertexIdx =  MAX_NUM_VERTEX * 2 * posIdx;
        KeyPoint keyPt;
        keyPt.classId = bestClassId;
        for (int clsIdx = 0; clsIdx < numClass; ++clsIdx)
        {
            keyPt.classScores.push_back(classScoreMap[startclsIdx + clsIdx]);
        }
        keyPt.scoreForSort = bestClassScore * centernessMap[posIdx]; // max_scores
        keyPt.gridcenter = meshgrid[posIdx];
        for (int i = 0; i < MAX_NUM_VERTEX; ++i)
        {
            keyPt.vertex[i].x = scale * regressionMap[vertexIdx + 2 * i];
            keyPt.vertex[i].y = scale * regressionMap[vertexIdx + 2 * i + 1];
        }
        keyPt.centerness = centernessMap[posIdx];
        keypoints.push_back(std::move(keyPt));
    }

    // topk, distance to vertex, separate them into each class
    std::vector<std::deque<KeyPoint>> classKeyPoints(numClass); // use deque, because its pop_front is the fastest

    std::partial_sort(keypoints.begin(), keypoints.begin() + topk, keypoints.end(),
        [](KeyPoint a, KeyPoint b) {return a.scoreForSort > b.scoreForSort;});

    for (int ordIdx = 0; ordIdx < topk; ++ordIdx)
    {
        for (int i = 0; i < MAX_NUM_VERTEX; ++i)
        {
            keypoints[ordIdx].vertex[i].x += keypoints[ordIdx].gridcenter.x;
            keypoints[ordIdx].vertex[i].y += keypoints[ordIdx].gridcenter.y; // vertexes
        }
        for (int clsIdx = 0; clsIdx < numClass; ++clsIdx)
        {
            // remove keypoints by class score threshold
            if (keypoints[ordIdx].classScores[clsIdx] > classScoreTh[clsIdx])
            {
                KeyPoint kpd;
                for (int i = 0; i < MAX_NUM_VERTEX; ++i)
                {
                    kpd.vertex[i] = keypoints[ordIdx].vertex[i];
                }
                kpd.vertexCenter = getAvgCenter(kpd);
                kpd.scoreForSort = keypoints[ordIdx].classScores[clsIdx] * keypoints[ordIdx].centerness;

                classKeyPoints[clsIdx].push_back(std::move(kpd));
            }
        }
    }

    // nms by distance for each class
    std::vector<std::vector<KeyPoint>> nmsKeyPoints(numClass);

    float squareNmsTh = nmsTh * nmsTh;
    for (int clsIdx = 0; clsIdx < numClass; ++clsIdx)
    {
        // remove keypoints by nms_threshold
        while (classKeyPoints[clsIdx].empty() != true)
        {
            // traverse from biggest score
            nmsKeyPoints[clsIdx].push_back(classKeyPoints[clsIdx].front());
            classKeyPoints[clsIdx].pop_front();

            int numRestKpt = classKeyPoints[clsIdx].size();
            std::deque<KeyPoint>::iterator it = classKeyPoints[clsIdx].begin();
            Coordinate frontvc = nmsKeyPoints[clsIdx].back().vertexCenter;

            // suppresion start from least score
            for (int i = numRestKpt - 1; i >= 0; --i)
            {
                Coordinate tmpvc = classKeyPoints[clsIdx][i].vertexCenter;
                float xdiff = frontvc.x - tmpvc.x;
                float ydiff = frontvc.y - tmpvc.y;
                float dist = xdiff * xdiff + ydiff * ydiff;
                if (dist < squareNmsTh)
                {
                    classKeyPoints[clsIdx].erase(it + i);
                }
            }
        }

        int numSuppressedKpt = nmsKeyPoints[clsIdx].size();
        for (int i = 0; i < numSuppressedKpt; ++i)
        {
            for (int j = 0; j < MAX_NUM_VERTEX; ++j)
            {
                nmsKeyPoints[clsIdx][i].vertex[j].x /= netW;
                nmsKeyPoints[clsIdx][i].vertex[j].y /= netH;
            }
        }
    }

    return nmsKeyPoints;
}