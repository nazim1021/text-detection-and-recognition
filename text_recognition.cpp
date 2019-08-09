#include <opencv2/text/ocr.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <cpprest/json.h>
#include <iostream>
#include <string>
#include <iomanip>
#include "decode.cpp"

using namespace web;
using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace cv::text;

const string keys =
    "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera.}"
    "{ model m     | | Path to a binary .pb file contains trained network.}"
    "{ detect d     | | to detect word or line.}"
    "{ width       | 800 | Preprocess input image by resizing to a specific width. It should be multiple by 32. }"
    "{ height      | 800 | Preprocess input image by resizing to a specific height. It should be multiple by 32. }"
    "{ thr         | 0.6 | Confidence threshold. }"
    "{ nms         | 0.4 | Non-maximum suppression threshold. }";

struct Triplets
{
    long unsigned int idx;
    vector<RotatedRect> box;
    vector<int> ind;
};

int main(int argc, char **argv)
{
    // configure tesseract
    Ptr<OCRTesseract> ocr = OCRTesseract::create(NULL, NULL, NULL, 3, 8);

    // Parse command line arguments.
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run TensorFlow implementation (https://github.com/argman/EAST) of "
                 "EAST: An Efficient and Accurate Scene Text Detector (https://arxiv.org/abs/1704.03155v2)");
    if (argc == 1)
    {
        parser.printMessage();
        return 0;
    }

    float confThreshold = parser.get<float>("thr");
    float nmsThreshold = parser.get<float>("nms");
    int inpWidth = parser.get<int>("width");
    int inpHeight = parser.get<int>("height");
    String model = parser.get<String>("model");
    String detect = parser.get<String>("detect");

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    CV_Assert(!model.empty());

    // Load network.
    Net net = readNetFromTensorflow(model);

    // get network output
    vector<Mat> outs;
    vector<String> outNames(2);
    outNames[0] = "feature_fusion/Conv_7/Sigmoid";
    outNames[1] = "feature_fusion/concat_3";

    static const std::string kWinName = "EAST: An Efficient and Accurate Scene Text Detector";
    namedWindow(kWinName, WINDOW_NORMAL);

    Mat frame, blob;
    if (parser.has("input"))
        frame = imread(parser.get<String>("input"), IMREAD_COLOR);

    blobFromImage(frame, blob, 1.0, Size(inpWidth, inpHeight), Scalar(123.68, 116.78, 103.94), true, false);
    net.setInput(blob);
    net.forward(outs, outNames);
    Mat scores = outs[0];
    Mat geometry = outs[1];

    vector<RotatedRect> boxes;
    vector<float> confidences;
    vector<vector<RotatedRect>> line_boxes;
    vector<vector<float>> line_confidences;
    decode(scores, geometry, confThreshold, boxes, line_boxes, confidences, line_confidences);

    int prev_size = 0;
    vector<int> indices;
    vector<Triplets> lines;
    //applying non maximum suppression
    for (int k = 0; k < line_boxes.size(); ++k)
    {
        int curr_size = line_boxes[k].size();
        if (curr_size == prev_size)
            continue;
        NMSBoxes(line_boxes[k], line_confidences[k], confThreshold, nmsThreshold, indices);
        lines.push_back({line_boxes[k].size(), line_boxes[k], indices});
        prev_size = curr_size;
    }

    int curr_max = 0;
    long unsigned int curr_idx;
    vector<RotatedRect> curr_box;
    vector<int> curr_ind;
    vector<Triplets> box_indices;
    for (int i = 0; i < lines.size(); i++)
    {
        if (lines[i].idx == 0)
        {
            box_indices.push_back({curr_idx, curr_box, curr_ind});
            curr_max = 0;
        }
        else
        {
            if (lines[i].idx > curr_max)
            {
                curr_idx = lines[i].idx;
                curr_box = lines[i].box;
                curr_ind = lines[i].ind;

                curr_max = lines[i].idx;
            }
        }
    }

    vector<json::value> imageText;

    for (int i = 0; i < box_indices.size(); i++)
    {
        vector<int> indices = box_indices[i].ind;
        vector<RotatedRect> line_boxes = box_indices[i].box;
        if (indices.size() != 0)
        {
            sort(indices.begin(), indices.end());
            // Render detections.
            Point2f ratio((float)frame.cols / inpWidth, (float)frame.rows / inpHeight);
            Point2f vertices[4];
            if (detect == "word")
            {
                for (size_t n = 0; n < indices.size(); ++n)
                {
                    RotatedRect &box = line_boxes[indices[n]];
                    box.points(vertices);

                    for (int j = 0; j < 4; ++j)
                    {
                        vertices[j].x *= ratio.x;
                        vertices[j].y *= ratio.y;
                    }

                    for (int j = 0; j < 4; ++j)
                        line(frame, vertices[j], vertices[(j + 1) % 4], Scalar(0, 255, 0), 1);
                }
            }
            else
            {
                RotatedRect &start_box = line_boxes[indices[0]];
                Point2f start_vertices[4];
                start_box.points(start_vertices);
                vertices[0] = start_vertices[0];
                vertices[1] = start_vertices[1];

                int last = indices.size() - 1;
                RotatedRect &end_box = line_boxes[indices[last]];
                Point2f end_vertices[4];
                end_box.points(end_vertices);
                vertices[2] = end_vertices[2];
                vertices[3] = end_vertices[3];

                for (int j = 0; j < 4; ++j)
                {
                    vertices[j].x *= ratio.x;
                    vertices[j].y *= ratio.y;
                }
                for (int j = 0; j < 4; ++j)
                    line(frame, vertices[j], vertices[(j + 1) % 4], Scalar(0, 255, 0), 1);
            }

            vector<Point2f> vert{{vertices[0].x, vertices[0].y},
                                 {vertices[1].x, vertices[1].y},
                                 {vertices[2].x, vertices[2].y},
                                 {vertices[3].x, vertices[3].y}};

            // Text recognition
            RotatedRect new_box = minAreaRect(vert);
            Rect rect3 = new_box.boundingRect();
            Mat box_img = frame(rect3);
            string output;
            ocr->run(box_img, output);
            output.erase(remove(output.begin(), output.end(), '\n'), output.end());

            //create json data
            json::value text_data;
            vector<json::value> box_data;
            for (int m = 0; m < 4; ++m)
            {
                json::value box_coord;
                stringstream xstream;
                xstream << fixed << setprecision(2) << vertices[m].x;
                string x = xstream.str();
                box_coord["x"] = json::value::string(x);
                stringstream ystream;
                ystream << fixed << setprecision(2) << vertices[m].y;
                string y = ystream.str();
                box_coord["y"] = json::value::string(y);

                box_data.push_back(box_coord);
            }

            text_data["text"] = json::value::string(output);
            text_data["boundingBox"] = json::value::array(box_data);
            imageText.push_back(text_data);
        }
    }

    for(int j = 0; j < imageText.size(); ++j){
        cout<<imageText[j];
        break;
    }
    // Put efficiency information.
    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    string label = format("Inference time: %.2f ms", t);
    putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
    imwrite("output.jpg", frame);
    imshow(kWinName, frame);
    waitKey(0);
    return 0;
}