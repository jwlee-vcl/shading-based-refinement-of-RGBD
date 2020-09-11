#pragma once

#include <pcl/point_types.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

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
#ifdef _DEBUG
    log_file_.open("lightestimator_log.txt");
#endif
  }

  void compute(const pcl::PointCloud<PointT>& cloud,
    pcl::PointCloud<NormalT>& normals,
    std::vector<size_t>& index,
    std::vector<float>& illum_coeffs,
    pcl::PointCloud<pcl::RGB>& albedo,
    pcl::PointCloud<pcl::RGB>& irrad);
  
  void compute_normals(
    const pcl::PointCloud<PointT>& cloud,
    pcl::PointCloud<NormalT>& normals);
  
  void filter_pixels_by_normal(
    const pcl::PointCloud<PointT>& cloud,
    const pcl::PointCloud<NormalT>& normals,
    std::vector<size_t>& index);

  void compute_illum_coeffs(
    const pcl::PointCloud<PointT>& cloud,
    const pcl::PointCloud<NormalT>& normals,
    const std::vector<size_t>& index,
    std::vector<float>& illum_coeffs,
    pcl::PointCloud<pcl::RGB>& albedo,
    pcl::PointCloud<pcl::RGB>& irrad);

private:
  double lambda_l_;

#ifdef _DEBUG
  std::fstream log_file_;
#endif // _DEBUG
  
};

