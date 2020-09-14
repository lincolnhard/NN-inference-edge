#include <assert.h>
#include <deque>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "postprocess_fcos.hpp"
#include "log_stream.hpp"


static auto LOG = spdlog::stdout_color_mt("FCOS");




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


bool PostprocessFCOS::suppressedByIOM(KeyPoint frontkpt, KeyPoint otherkpt, float th)
{
    auto ax1 = frontkpt.vertexTL.x;
    auto ax2 = frontkpt.vertexBR.x;
    auto ay1 = frontkpt.vertexTL.y;
    auto ay2 = frontkpt.vertexBR.y;
    auto bx1 = otherkpt.vertexTL.x;
    auto bx2 = otherkpt.vertexBR.x;
    auto by1 = otherkpt.vertexTL.y;
    auto by2 = otherkpt.vertexBR.y;

    auto w = overlap(ax1, ax2, bx1, bx2);
    auto h = overlap(ay1, ay2, by1, by2);
    if (w < 0 || h < 0)
    {
        return false;
    }
    auto interArea = w * h;
    auto minArea = std::min(((ax2 - ax1 + 1) * (ay2 - ay1 + 1)), ((bx2 - bx1 + 1) * (by2 - by1 + 1)));
    auto iou = interArea / minArea;
    return iou > th;
}


bool PostprocessFCOS::suppressedByIOU(KeyPoint frontkpt, KeyPoint otherkpt, float th)
{
    auto ax1 = frontkpt.vertexTL.x;
    auto ax2 = frontkpt.vertexBR.x;
    auto ay1 = frontkpt.vertexTL.y;
    auto ay2 = frontkpt.vertexBR.y;
    auto bx1 = otherkpt.vertexTL.x;
    auto bx2 = otherkpt.vertexBR.x;
    auto by1 = otherkpt.vertexTL.y;
    auto by2 = otherkpt.vertexBR.y;

    auto w = overlap(ax1, ax2, bx1, bx2);
    auto h = overlap(ay1, ay2, by1, by2);
    if (w < 0 || h < 0)
    {
        return false;
    }
    auto interArea = w * h;
    auto unionArea = (ax2 - ax1 + 1) * (ay2 - ay1 + 1) + (bx2 - bx1 + 1) * (by2 - by1 + 1) - interArea;
    auto iou = interArea / unionArea;
    return iou > th;
}


bool PostprocessFCOS::suppressedByDist(KeyPoint frontkpt, KeyPoint otherkpt, float th)
{
    Coordinate othervc = otherkpt.vertexCenter;
    Coordinate frontvc = frontkpt.vertexCenter;
    float xdiff = frontvc.x - othervc.x;
    float ydiff = frontvc.y - othervc.y;
    float dist = xdiff * xdiff + ydiff * ydiff;
    float squareNmsTh = th * th;
    return dist < squareNmsTh;
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
    numClass = config["num_class_bbox"].get<int>();
    // nmsTh = config["nms_threshold_dist"].get<float>();
    nmsTh = config["nms_threshold_iou"].get<float>();
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

    for (int clsIdx = 0; clsIdx < numClass; ++clsIdx)
    {
        while (classKeyPoints[clsIdx].empty() != true)
        {
            // traverse from biggest score
            nmsKeyPoints[clsIdx].push_back(classKeyPoints[clsIdx].front());
            classKeyPoints[clsIdx].pop_front();

            int numRestKpt = classKeyPoints[clsIdx].size();
            auto frontkpt = nmsKeyPoints[clsIdx].back();
            // suppresion start from least score
            for (int i = numRestKpt - 1; i >= 0; --i)
            {
                auto otherkpt = classKeyPoints[clsIdx][i];

                // if (suppressedByDist(frontkpt, otherkpt, nmsTh))
                // if (suppressedByIOU(frontkpt, otherkpt, nmsTh))
                if (suppressedByIOM(frontkpt, otherkpt, nmsTh))
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
