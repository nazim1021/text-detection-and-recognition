# Text Detection and Recognition 

This project focusses on using OpenCV to detect text in images using the EAST text detector. The bounding box be obtained for individual texts as well as lines. Once the text is detected, we then use tesseract C++ api to recognize/extract the detected text. 


## REQUIREMENTS
1. Ubuntu 16.04/18.04
2. C++ (g++ compiler) 
3. [Tesseract](https://github.com/tesseract-ocr/tesseract/wiki/Compiling-%E2%80%93-GitInstallation)
4. [OpenCV](https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/)


## USAGE
Run the file text_recognition.cpp as below

```g++ text_recognition.cpp -lboost_system -lcrypto -lssl -lcpprest `pkg-config --cflags --libs tesseract opencv` ```

``` ./a.out -i=sample.jpg -m=frozen_east_text_detection.pb -d=line ```

## RESULTS
Text detection output:

<img src="output.jpg" width="300">

Text recognition output:

`{
    "text": "HEALTHY FOOD MENU!",
    "boundingBox": [
        {
            "x": "280.38",
            "y": "62.87"
        },
        {
            "x": "280.13",
            "y": "29.76"
        },
        {
            "x": "686.98",
            "y": "27.60"
        },
        {
            "x": "687.45",
            "y": "62.72"
        }
    ]
}`

## REFERENCES
- [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155)
- https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
- https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.cpp
- https://github.com/opencv/opencv_contrib/blob/master/modules/text/samples/end_to_end_recognition.cpp
