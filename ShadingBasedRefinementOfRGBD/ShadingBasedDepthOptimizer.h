#pragma once

#include "ImagePyramid.h"
#include "LightEstimator.h"
#include "DepthOptimizer.h"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

struct OptimizerSettings
{
  // blur
  bool apply_blur = true;  
  int blur_size = 0;
  float sigma_color = 0.03f;
  float sigma_space = 2.0f;

  // camera param
  cv::Point2f focal = cv::Point2f(574.053f, 574.053f);
  cv::Point2f pp = cv::Point2f(320.0f, 240.0f);

  // depth optimizer
  bool use_multigrid = true;
  size_t num_scale = 3;

  size_t num_iter = 1;

  // weights
  float w_grad = 0.1f;
  float w_smooth = 400.0f;
  float w_depth = 10.0f;
  float w_temp = 0.0f; 
};

class ShadingBasedDepthOptimizer
{
public:
  ShadingBasedDepthOptimizer(OptimizerSettings& settings);
  ~ShadingBasedDepthOptimizer();

  void set_data(const cv::Mat& grayf, const cv::Mat& depthf);

  void compute();

  void depth_optimize(
    const cv::Mat& validmap,
    const std::vector<float>& illum_coeffs, cv::Mat& albedof);

  void depth_optimize_multigrid(
    const cv::Mat& validmap,
    const std::vector<float>& illum_coeffs, cv::Mat& albedof);

  cv::Mat get_refined_depth()
  {
    return refined_depthf_.clone();
  }

  static void visualize_depth(const cv::Mat& depthf,
    const cv::Point2f& focal, const cv::Point2f& pp,
    const std::string& name = "depth");
  
  static void depth2point(
    const cv::Mat& depthf,
    pcl::PointCloud<pcl::PointXYZ>& cloud,
    const cv::Point2f& focal, const cv::Point2f& pp);
  
protected:
  OptimizerSettings settings_;

  cv::Mat grayf_;
  cv::Mat initial_depthf_;

  cv::Mat albedo_grayf_;

  cv::Mat refined_depthf_;
};

