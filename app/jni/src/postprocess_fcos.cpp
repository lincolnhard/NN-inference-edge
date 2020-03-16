#include <assert.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "log_stream.hpp"
#include "postprocess_fcos.hpp"
#include <deque>

static auto LOG = spdlog::stdout_color_mt("FCOS");

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
    classScoreTh = config["class_score_threshold"].get<float>();

    assert(netW == featW * stride);
    assert(netH == featH * stride);

    initMeshgrid();
    SLOG_INFO << "Meshgrid initialization done" << std::endl;
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

void PostprocessFCOS::setInput(std::vector<float *> featuremaps)
{
    classScoreMap = featuremaps[0];
    centernessMap = featuremaps[1];
    regressionMap = featuremaps[2];
}

Coordinate PostprocessFCOS::getAvgCenter(KeyPoint kpt)
{
    Coordinate avgpt;
    avgpt.x = 0.25f * (kpt.vertex[0].x + kpt.vertex[1].x + kpt.vertex[2].x + kpt.vertex[3].x);
    avgpt.y = 0.25f * (kpt.vertex[0].y + kpt.vertex[1].y + kpt.vertex[2].y + kpt.vertex[3].y);
    return avgpt;
}

void PostprocessFCOS::run(std::vector<std::vector<ScoreVertices>> &predResult)
{
    // score argmax
    std::vector<KeyPoint> keypoints;
    const int FEAT_PLANE_SIZE = featW * featH;
    SLOG_INFO << "FEAT_PLANE_SIZE: " << FEAT_PLANE_SIZE << std::endl; 
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
        keyPt.classScore = bestClassScore;
        keyPt.score = bestClassScore * centernessMap[posIdx];
        keyPt.center = meshgrid[posIdx];
        keyPt.vertex[0].x = scale * regressionMap[vertexIdx];
        keyPt.vertex[0].y = scale * regressionMap[vertexIdx + 1];
        keyPt.vertex[1].x = scale * regressionMap[vertexIdx + 2];
        keyPt.vertex[1].y = scale * regressionMap[vertexIdx + 3];
        keyPt.vertex[2].x = scale * regressionMap[vertexIdx + 4];
        keyPt.vertex[2].y = scale * regressionMap[vertexIdx + 5];
        keyPt.vertex[3].x = scale * regressionMap[vertexIdx + 6];
        keyPt.vertex[3].y = scale * regressionMap[vertexIdx + 7];
        keypoints.push_back(std::move(keyPt));
    }

    SLOG_INFO << "score argmax OK" << std::endl;

    // topk, distance to vertex, separate them into each class
    std::vector<std::vector<KeyPoint>> classKeyPoints(numClass);
    std::partial_sort(keypoints.begin(), keypoints.begin() + topk, keypoints.end(), [](KeyPoint a, KeyPoint b) {return a.score > b.score;});
    for (int ordIdx = 0; ordIdx < topk; ++ordIdx)
    {
        int clsIdx = keypoints[ordIdx].classId;
        keypoints[ordIdx].vertex[0].x = keypoints[ordIdx].center.x - keypoints[ordIdx].vertex[0].x;
        keypoints[ordIdx].vertex[0].y = keypoints[ordIdx].center.y - keypoints[ordIdx].vertex[0].y;
        keypoints[ordIdx].vertex[1].x = keypoints[ordIdx].center.x + keypoints[ordIdx].vertex[1].x;
        keypoints[ordIdx].vertex[1].y = keypoints[ordIdx].center.y - keypoints[ordIdx].vertex[1].y;
        keypoints[ordIdx].vertex[2].x = keypoints[ordIdx].center.x + keypoints[ordIdx].vertex[2].x;
        keypoints[ordIdx].vertex[2].y = keypoints[ordIdx].center.y + keypoints[ordIdx].vertex[2].y;
        keypoints[ordIdx].vertex[3].x = keypoints[ordIdx].center.x - keypoints[ordIdx].vertex[3].x;
        keypoints[ordIdx].vertex[3].y = keypoints[ordIdx].center.y + keypoints[ordIdx].vertex[3].y;
        classKeyPoints[clsIdx].push_back(std::move(keypoints[ordIdx]));
    }

    SLOG_INFO << "topk OK" << std::endl;

    // nms
    std::vector<std::deque<KeyPoint>> tmpKeypoints(numClass);  // use deque, because its pop_front is the fastest
    std::vector<std::vector<KeyPoint>> nmsKeyPoints(numClass);

    float squareNmsTh = nmsTh * nmsTh;
    for (int clsIdx = 0; clsIdx < numClass; ++clsIdx)
    {
        if (classKeyPoints[clsIdx].empty() != true)
        {
            // remove keypoints by class_score_threshold
            for (KeyPoint kpd: classKeyPoints[clsIdx])
            {
                if (kpd.classScore > classScoreTh)
                {
                    kpd.vertexCenter = getAvgCenter(kpd);
                    tmpKeypoints[clsIdx].push_back(std::move(kpd));
                }
            }

            // remove keypoints by nms_threshold
            while (tmpKeypoints[clsIdx].empty() != true)
            {
                // traverse from biggest score
                nmsKeyPoints[clsIdx].push_back(tmpKeypoints[clsIdx].front());
                tmpKeypoints[clsIdx].pop_front();

                int numRestKpt = tmpKeypoints[clsIdx].size();
                std::deque<KeyPoint>::iterator it = tmpKeypoints[clsIdx].begin();
                Coordinate frontvc = nmsKeyPoints[clsIdx].back().vertexCenter;

                // suppresion start from least score
                for (int i = numRestKpt - 1; i >= 0; --i)
                {
                    Coordinate tmpvc = tmpKeypoints[clsIdx][i].vertexCenter;
                    float xdiff = frontvc.x - tmpvc.x;
                    float ydiff = frontvc.y - tmpvc.y;
                    float dist = xdiff * xdiff + ydiff * ydiff;
                    if (dist < squareNmsTh)
                    {
                        tmpKeypoints[clsIdx].erase(it + i);
                    }
                }
            }

            for (int i = 0; i < nmsKeyPoints[clsIdx].size(); ++i)
            {
                nmsKeyPoints[clsIdx][i].vertex[0].x = nmsKeyPoints[clsIdx][i].vertex[0].x / netW;
                nmsKeyPoints[clsIdx][i].vertex[0].y = nmsKeyPoints[clsIdx][i].vertex[0].y / netH;
                nmsKeyPoints[clsIdx][i].vertex[1].x = nmsKeyPoints[clsIdx][i].vertex[1].x / netW;
                nmsKeyPoints[clsIdx][i].vertex[1].y = nmsKeyPoints[clsIdx][i].vertex[1].y / netH;
                nmsKeyPoints[clsIdx][i].vertex[2].x = nmsKeyPoints[clsIdx][i].vertex[2].x / netW;
                nmsKeyPoints[clsIdx][i].vertex[2].y = nmsKeyPoints[clsIdx][i].vertex[2].y / netH;
                nmsKeyPoints[clsIdx][i].vertex[3].x = nmsKeyPoints[clsIdx][i].vertex[3].x / netW;
                nmsKeyPoints[clsIdx][i].vertex[3].y = nmsKeyPoints[clsIdx][i].vertex[3].y / netH;
            }

        }
    }

    SLOG_INFO << "nms OK" << std::endl;

    // fill in predResult
    for (int clsIdx = 0; clsIdx < numClass; ++clsIdx)
    {
        // maybe not clear result here
        if (!predResult[clsIdx].empty())
        {
            predResult[clsIdx].clear();
        }

        // maybe not show info here
        int numClsObj = nmsKeyPoints[clsIdx].size();
        SLOG_INFO << "class " << clsIdx << ": " << numClsObj << std::endl;

        for (int i = 0; i < numClsObj; ++i)
        {
            ScoreVertices sv;
            sv.x0 = nmsKeyPoints[clsIdx][i].vertex[0].x;
            sv.y0 = nmsKeyPoints[clsIdx][i].vertex[0].y;
            sv.x1 = nmsKeyPoints[clsIdx][i].vertex[1].x;
            sv.y1 = nmsKeyPoints[clsIdx][i].vertex[1].y;
            sv.x2 = nmsKeyPoints[clsIdx][i].vertex[2].x;
            sv.y2 = nmsKeyPoints[clsIdx][i].vertex[2].y;
            sv.x3 = nmsKeyPoints[clsIdx][i].vertex[3].x;
            sv.y3 = nmsKeyPoints[clsIdx][i].vertex[3].y;
            sv.score = nmsKeyPoints[clsIdx][i].score;
            predResult[clsIdx].push_back(std::move(sv));
        }
    }

    SLOG_INFO << "fillin predResult OK" << std::endl;
}