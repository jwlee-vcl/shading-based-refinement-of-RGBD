#pragma once

#include <pcl/point_types.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/opencv.hpp>

#include <fstream>

#ifndef PointT
typedef pcl::PointXYZRGB PointT;
typedef pcl::Normal NormalT;
#endif // PointT

class LightingEstimator
{
public:
  LightingEstimator(double lambda_l = 10)
    :lambda_l_(lambda_l)
  {
//#ifdef _DEBUG
//    log_file_.open("lightestimator_log.txt");
//#endif
  }

  void compute(
    const cv::Mat& depthf,
    const cv::Mat& grayf,
    std::vector<float>& illum_coeffs,
    cv::Mat& albedof,
    const cv::Point2f& focal,
    const cv::Point2f& pp);
  
  void compute_normals(
    const cv::Mat& depthf,
    cv::Mat& points,
    cv::Mat& normals,
    const cv::Point2f& inv_focal,
    const cv::Point2f& pp);
  
  void filter_pixels_by_normal(
    const cv::Mat& points,
    const cv::Mat& normals,
    std::vector<size_t>& index);
    
  void compute_illum_coeffs(
    const cv::Mat& grayf,
    const cv::Mat& normals,
    const std::vector<size_t>& index,
    std::vector<float>& illum_coeffs,
    cv::Mat& albedo_grayf);

  void depth2point(
    const cv::Mat& depthf,
    cv::Mat& points,
    const cv::Point2f& inv_focal,
    const cv::Point2f& pp);

  void compute_sphere_normal(cv::Mat& normals, int dim);

  void compute_lightmap(
    cv::Mat& lightmap,
    const cv::Mat& normals, const std::vector<float>& illum_coeffs);

  cv::Mat get_validmap()
  {
    return validmap_.clone();
  }

protected:
  float lambda_l_;

  cv::Mat radf_;  
  cv::Mat irradf_;
  
  cv::Mat validmap_;

//#ifdef _DEBUG
//  std::fstream log_file_;
//#endif // _DEBUG
  
};