#pragma once

#ifdef _DEBUG

#pragma comment(lib, "opencv_core300d.lib")
#pragma comment(lib, "opencv_imgproc300d.lib")
#pragma comment(lib, "opencv_highgui300d.lib")
#pragma comment(lib, "opencv_videoio300d.lib")
#pragma comment(lib, "opencv_imgcodecs300d.lib")
#pragma comment(lib, "opencv_calib3d300d.lib")
#pragma comment(lib, "opencv_features2d300d.lib")
#pragma comment(lib, "opencv_objdetect300d.lib")
#pragma comment(lib, "opencv_video300d.lib")
#pragma comment(lib, "opencv_ml300d.lib")

#else // _DEBUG

#pragma comment(lib, "opencv_core300.lib")
#pragma comment(lib, "opencv_imgproc300.lib")
#pragma comment(lib, "opencv_highgui300.lib")
#pragma comment(lib, "opencv_videoio300.lib")
#pragma comment(lib, "opencv_imgcodecs300.lib")
#pragma comment(lib, "opencv_calib3d300.lib")
#pragma comment(lib, "opencv_features2d300.lib")
#pragma comment(lib, "opencv_objdetect300.lib")
#pragma comment(lib, "opencv_video300.lib")
#pragma comment(lib, "opencv_ml300.lib")

#endif // _DEBUG
