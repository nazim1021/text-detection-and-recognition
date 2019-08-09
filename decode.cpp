#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

//get bounding boxes from predictions
static void decode(const Mat &scores, const Mat &geometry, float scoreThresh,
            vector<RotatedRect> &detections, vector<vector<RotatedRect>> &line_detections,
            vector<float> &confidences, vector<vector<float>> &box_confidences)
{
    detections.clear();
    confidences.clear();
    CV_Assert(scores.dims == 4);
    CV_Assert(geometry.dims == 4);
    CV_Assert(scores.size[0] == 1);
    CV_Assert(geometry.size[0] == 1);
    CV_Assert(scores.size[1] == 1);
    CV_Assert(geometry.size[1] == 5);
    CV_Assert(scores.size[2] == geometry.size[2]);
    CV_Assert(scores.size[3] == geometry.size[3]);

    const int height = scores.size[2];
    const int width = scores.size[3];
    for (int y = 0; y < height; ++y)
    {
        vector<RotatedRect> boxes;
        vector<float> conf;
        const float *scoresData = scores.ptr<float>(0, 0, y);
        const float *x0_data = geometry.ptr<float>(0, 0, y);
        const float *x1_data = geometry.ptr<float>(0, 1, y);
        const float *x2_data = geometry.ptr<float>(0, 2, y);
        const float *x3_data = geometry.ptr<float>(0, 3, y);
        const float *anglesData = geometry.ptr<float>(0, 4, y);
        for (int x = 0; x < width; ++x)
        {
            float score = scoresData[x];
            if (score < scoreThresh)
                continue;

            // Decode a prediction.
            // Multiple by 4 because feature maps are 4 time less than input image.
            float offsetX = x * 4.0f, offsetY = y * 4.0f;
            float angle = anglesData[x];
            float cosA = std::cos(angle);
            float sinA = std::sin(angle);
            float h = x0_data[x] + x2_data[x];
            float w = x1_data[x] + x3_data[x];

            Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                           offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
            Point2f p1 = Point2f(-sinA * h, -cosA * h) + offset;
            Point2f p3 = Point2f(-cosA * w, sinA * w) + offset;
            RotatedRect r(0.5f * (p1 + p3), Size2f(w, h), -angle * 180.0f / (float)CV_PI);

            boxes.push_back(r);
            detections.push_back(r);
            conf.push_back(score);
            confidences.push_back(score);

        }
        line_detections.push_back(boxes);
        box_confidences.push_back(conf);
    }
}