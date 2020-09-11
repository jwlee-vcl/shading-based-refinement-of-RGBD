#include "configs_pcl.h"
#include "configs_opencv.h"

#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingOpenGL);

#include <pcl/io/openni2_grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include "ShadingBasedDepthOptimizer.h"

#include <pcl/range_image/range_image.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/surface/organized_fast_mesh.h>

#include <opencv2/opencv.hpp>

#include <Eigen/Sparse>

#include <ctime>

//#ifdef _DEBUG
std::ofstream log_file;
//#endif

int main()
{
  std::cout << "load depth map" << std::endl;
  cv::Mat color = cv::imread("frame-000000.color.png", cv::IMREAD_UNCHANGED);
  cv::Mat depth = cv::imread("frame-000000.depth.png", cv::IMREAD_UNCHANGED);

  std::cout << "number of color channels: " << color.channels() << std::endl;
  std::cout << "number of color depth: " << color.depth() << std::endl;

  std::cout << "number of depth channels: " << depth.channels() << std::endl;
  std::cout << "number of depth depth: " << depth.depth() << std::endl;

  int width = depth.cols;
  int height = depth.rows;

  cv::Point2f focal(574.053f, 574.053f);
  cv::Point2f pp(320.0f, 240.0f);

  // resize image to [640, 480]
  cv::Mat color_resized;
  cv::resize(color, color_resized, cv::Size(width, height));

  cv::Mat gray;
  cv::cvtColor(color_resized, gray, cv::COLOR_BGR2GRAY);
  
  cv::Mat grayf;
  gray.convertTo(grayf, CV_32F, 1.0f/255.0f);

  cv::Mat depthf;
  depth.convertTo(depthf, CV_32F, 0.001f); // mm to m
  
  OptimizerSettings settings;
  settings.apply_blur     = true;
  settings.blur_size      = 0;
  settings.sigma_color    = 0.03f;
  settings.sigma_space    = 2.0f;
  settings.focal          = cv::Point2f(574.053f, 574.053f);
  settings.pp             = cv::Point2f(320.0f, 240.0f);  
  settings.use_multigrid  = true;
  settings.num_scale      = 3;  
  settings.num_iter       = 1;  
  settings.w_grad         = 0.1f;
  settings.w_smooth       = 5000.0f;
  settings.w_depth        = 10.0f;
  settings.w_temp         = 0.0f;

  ShadingBasedDepthOptimizer optimizer(settings);
  optimizer.set_data(grayf, depthf);
  optimizer.compute();

  cv::Mat depth_refined = optimizer.get_refined_depth();

  ShadingBasedDepthOptimizer::visualize_depth(depth_refined, focal, pp, "refined");

  return 0;
}