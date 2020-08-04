#include <assert.h>
#include <deque>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "postprocess_fcos.hpp"
#include "log_stream.hpp"


static auto LOG = spdlog::stdout_color_mt("FCOS");

// Coordinate PostprocessFCOS::getAvgCenter(KeyPoint kpt)
// {
//     Coordinate avgpt;
//     avgpt.x = 0.5f * (kpt.vertex[0].x + kpt.vertex[1].x);
//     avgpt.y = 0.5f * (kpt.vertex[0].y + kpt.vertex[1].y);
//     return avgpt;
// }


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
    nmsTh = config["nms_threshold"].get<float>();
    classScoreTh = config["class_score_threshold"].get<std::vector<float> >();

    assert(netW == featW * stride);
    assert(netH == featH * stride);
    assert(classScoreTh.size() == numClass);

    initMeshgrid();
}

std::vector<std::vector<KeyPoint>> PostprocessFCOS::run(std::vector<const float *> &featuremaps)
{
    const float *classScoreMap = featuremaps[0];
    const float *centernessMap = featuremaps[1];
    const float *regressionMap = featuremaps[2];

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


        KeyPoint keyPt;
        keyPt.classId = bestClassId;
        for (int clsIdx = 0; clsIdx < numClass; ++clsIdx)
        {
            keyPt.classScores.push_back(classScoreMap[startclsIdx + clsIdx]);
        }
        keyPt.scoreForSort = bestClassScore * centernessMap[posIdx];
        keyPt.gridcenter = meshgrid[posIdx];
        int vtxIdx = 4 * posIdx; // xmin, ymin, xmax, ymax
        keyPt.vertexTL.x = regressionMap[vtxIdx];
        keyPt.vertexTL.y = regressionMap[vtxIdx + 1];
        keyPt.vertexBR.x = regressionMap[vtxIdx + 2];
        keyPt.vertexBR.y = regressionMap[vtxIdx + 3];
        keyPt.centerness = centernessMap[posIdx];
        keypoints.push_back(std::move(keyPt));
    }


    // topk sort
    std::partial_sort(keypoints.begin(), keypoints.begin() + topk, keypoints.end(),
        [](KeyPoint a, KeyPoint b) {return a.scoreForSort > b.scoreForSort;});


    // to bbox, separate them into each class
    std::vector<std::deque<KeyPoint>> classKeyPoints(numClass);
    // std::vector<std::vector<KeyPoint>> classKeyPoints(numClass);
    for (int ordIdx = 0; ordIdx < topk; ++ordIdx)
    {
        keypoints[ordIdx].vertexTL.x = keypoints[ordIdx].gridcenter.x - keypoints[ordIdx].vertexTL.x;
        keypoints[ordIdx].vertexTL.y = keypoints[ordIdx].gridcenter.y - keypoints[ordIdx].vertexTL.y;
        keypoints[ordIdx].vertexBR.x = keypoints[ordIdx].gridcenter.x + keypoints[ordIdx].vertexBR.x;
        keypoints[ordIdx].vertexBR.y = keypoints[ordIdx].gridcenter.y + keypoints[ordIdx].vertexBR.y;
        // TODO: clamping
        for (int clsIdx = 0; clsIdx < numClass; ++clsIdx)
        {
            // remove keypoints by class score threshold
            if (keypoints[ordIdx].classScores[clsIdx] > classScoreTh[clsIdx])
            {
                KeyPoint kpd;
                kpd.classId = keypoints[ordIdx].classId;
                kpd.vertexTL = keypoints[ordIdx].vertexTL;
                kpd.vertexBR = keypoints[ordIdx].vertexBR;
                kpd.vertexCenter.x = 0.5f * (kpd.vertexTL.x + kpd.vertexBR.x);
                kpd.vertexCenter.y = 0.5f * (kpd.vertexTL.y + kpd.vertexBR.y);
                kpd.scoreForSort = keypoints[ordIdx].classScores[clsIdx] * keypoints[ordIdx].centerness;
                classKeyPoints[clsIdx].push_back(std::move(kpd));
            }
        }
    }

    // nms by bbox center distance, class independent
    std::vector<std::vector<KeyPoint>> nmsKeyPoints(numClass);
    float squareNmsTh = nmsTh * nmsTh;

    for (int clsIdx = 0; clsIdx < numClass; ++clsIdx)
    {
        while (classKeyPoints[clsIdx].empty() != true)
        {
            // traverse from biggest score
            nmsKeyPoints[clsIdx].push_back(classKeyPoints[clsIdx].front());
            classKeyPoints[clsIdx].pop_front();
            // classKeyPoints[clsIdx].erase(classKeyPoints[clsIdx].begin());

            int numRestKpt = classKeyPoints[clsIdx].size();

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
                    classKeyPoints[clsIdx].erase(classKeyPoints[clsIdx].begin() + i);
                }
            }
        }

        int numSuppressedKpt = nmsKeyPoints[clsIdx].size();
        for (int i = 0; i < numSuppressedKpt; ++i)
        {
            nmsKeyPoints[clsIdx][i].vertexTL.x /= netW;
            nmsKeyPoints[clsIdx][i].vertexBR.x /= netW;
            nmsKeyPoints[clsIdx][i].vertexTL.y /= netH;
            nmsKeyPoints[clsIdx][i].vertexBR.y /= netH;
        }
    }

    return nmsKeyPoints;
}
