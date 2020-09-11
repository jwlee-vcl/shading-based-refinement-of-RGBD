#include "LightEstimator.h"

#include <pcl/filters/fast_bilateral_omp.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>

void LightingEstimator::compute(
  const pcl::PointCloud<PointT>& cloud, 
  pcl::PointCloud<NormalT>& normals, 
  std::vector<size_t>& index, 
  std::vector<float>& illum_coeffs, 
  pcl::PointCloud<pcl::RGB>& albedo,
  pcl::PointCloud<pcl::RGB>& irrad)
{
  size_t width = cloud.width;
  size_t height = cloud.height;
  
  std::cout << "estimate normals" << std::endl;
  compute_normals(cloud, normals);

  std::cout << "filter out pixels by angle" << std::endl;
  filter_pixels_by_normal(cloud, normals, index);
    
  std::cout << "compute illumination coefficients" << std::endl;
  
  compute_illum_coeffs(cloud, normals, index, illum_coeffs, albedo, irrad);

#ifdef _DEBUG
  pcl::PointCloud<pcl::RGB> filtered;
  pcl::copyPointCloud(cloud, filtered);

  for (size_t i = 0; i < index.size(); ++i)
  {
    filtered.points[index[i]].r = 255;
    //filtered.points[index[i]].g = 255;
    //filtered.points[index[i]].b = 255;
  }

  pcl::PointCloud<PointT>::Ptr cloud_ptr(new pcl::PointCloud<PointT>(cloud));
  pcl::PointCloud<NormalT>::Ptr normals_ptr(new pcl::PointCloud<NormalT>(normals));

  // visualize normals
  pcl::visualization::PCLVisualizer viewer("Point Cloud Viewer");
  viewer.setBackgroundColor(0.0, 0.0, 0.0);
  viewer.addPointCloud<PointT>(cloud_ptr, "cloud");
  viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
  viewer.addPointCloudNormals<PointT, NormalT>(cloud_ptr, normals_ptr, 100, 0.05f, "normals");
  viewer.addCoordinateSystem(0.1);
  viewer.initCameraParameters();

  pcl::visualization::ImageViewer image_viewer("Image Viewer");
  image_viewer.addRGBImage(cloud, "image");

  pcl::visualization::ImageViewer filtered_viewer("filtered Viewer");
  image_viewer.addRGBImage(filtered, "filtered image");

  pcl::visualization::ImageViewer albedo_viewer("Albedo Viewer");
  albedo_viewer.addRGBImage(albedo, "albedo image");

  pcl::visualization::ImageViewer irrad_viewer("Irradiance Viewer");
  irrad_viewer.addRGBImage(irrad, "irrad image");

  while (!viewer.wasStopped() && !image_viewer.wasStopped() &&
    !filtered_viewer.wasStopped() && !albedo_viewer.wasStopped() && !irrad_viewer.wasStopped())
  {
    viewer.spinOnce();
    image_viewer.spinOnce();
    //filtered_viewer.spinOnce();
    albedo_viewer.spinOnce();
    irrad_viewer.spinOnce();
  }
#endif
}

void LightingEstimator::compute_normals(
  const pcl::PointCloud<PointT>& cloud, 
  pcl::PointCloud<NormalT>& normals)
{  
  pcl::PointCloud<PointT>::Ptr cloud_ptr(new pcl::PointCloud<PointT>(cloud));

  // smoothing  
  pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);

  pcl::FastBilateralFilterOMP<PointT> bilateral_filter;
  bilateral_filter.setNumberOfThreads(8);
  bilateral_filter.setInputCloud(cloud_ptr);
  
  bilateral_filter.applyFilter(*cloud_filtered);

  // estimate normals      
  pcl::IntegralImageNormalEstimation<PointT, NormalT> ne;
  ne.setNormalEstimationMethod(ne.SIMPLE_3D_GRADIENT);
  ne.setMaxDepthChangeFactor(0.02f);
  ne.setNormalSmoothingSize(10.0f);
  ne.setInputCloud(cloud_filtered);

  ne.compute(normals);  
}

void LightingEstimator::filter_pixels_by_normal(
  const pcl::PointCloud<PointT>& cloud, 
  const pcl::PointCloud<NormalT>& normals, 
  std::vector<size_t>& index)
{
  index.resize(cloud.size());
  size_t j = 0;

  const static float angle_thresh = std::cosf(78.0 * M_PI / 180.0);
  std::cout << "angle threshold " << angle_thresh << std::endl;

  for (size_t i = 0; i < normals.size(); ++i)
  {
    if (!pcl_isfinite(normals.points[i].normal_x) ||
      !pcl_isfinite(normals.points[i].normal_y) ||
      !pcl_isfinite(normals.points[i].normal_z))
    {
      continue;
    }

    // compare angle
    Eigen::Vector3f view_dir = -cloud.points[i].getVector3fMap();
    view_dir.normalize();

    Eigen::Vector3f normal = normals.points[i].getNormalVector3fMap();

    float angle = view_dir.dot(normal);

    if (angle > angle_thresh)
    {
      index[j] = i;
      ++j;
    }
  }
  index.resize(j);

  std::cout << "number of valid pixels: " << index.size() << std::endl;
}

void LightingEstimator::compute_illum_coeffs(
  const pcl::PointCloud<PointT>& cloud, 
  const pcl::PointCloud<NormalT>& normals, 
  const std::vector<size_t>& index, 
  std::vector<float>& illum_coeffs, 
  pcl::PointCloud<pcl::RGB>& albedo, 
  pcl::PointCloud<pcl::RGB>& irrad)
{
  std::cout << "build linear system" << std::endl;

  // build matrix
  // A l = I
  Eigen::MatrixXf A(index.size(), 9);
  Eigen::MatrixXf I(index.size(), 1);

  static float color_denom = 1.0 / (3.0 * 255.0);

  for (size_t i = 0; i< index.size(); ++i)
  {
    Eigen::Vector3f n = normals.points[index[i]].getNormalVector3fMap();
    Eigen::Vector3i c = cloud.points[index[i]].getRGBVector3i();

    A(i, 0) = 1.0;
    A(i, 1) = n.y();
    A(i, 2) = n.z();
    A(i, 3) = n.x();
    A(i, 4) = n.x()*n.y();
    A(i, 5) = n.y()*n.z();
    A(i, 6) = -(n.x()*n.x()) - (n.y()*n.y()) + 2.0*(n.z()*n.z());
    A(i, 7) = n.z()*n.x();
    A(i, 8) = n.x()*n.x() - n.y()*n.y();

    I(i, 0) = (c.x() + c.y() + c.z()) * color_denom;
  }

#ifdef _DEBUG
  //log_file << "A = " << A << std::endl;
  //log_file << "I = " << I << std::endl;
#endif // _DEBUG 

  Eigen::MatrixXf AtA = A.transpose() * A;
  Eigen::MatrixXf AtI = A.transpose() * I;

  std::cout << "solve linear system" << std::endl;

  Eigen::MatrixXf l = AtA.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(AtI);
  
  Eigen::MatrixXf Al = A * l;

#ifdef _DEBUG
  //log_file_ << "l = " << l.transpose() << std::endl;
  //log_file_ << "Al = " << Al << std::endl;
#endif // _DEBUG 

  illum_coeffs.resize(9);
  illum_coeffs[0] = l(0);
  illum_coeffs[1] = l(1);
  illum_coeffs[2] = l(2);
  illum_coeffs[3] = l(3);
  illum_coeffs[4] = l(4);
  illum_coeffs[5] = l(5);
  illum_coeffs[6] = l(6);
  illum_coeffs[7] = l(7);
  illum_coeffs[8] = l(8);

  // compute albedo image
  std::cout << "compute albedo image" << std::endl;

  albedo.width = cloud.width;
  albedo.height = cloud.height;
  albedo.resize(albedo.width * albedo.height);

  irrad.width = cloud.width;
  irrad.height = cloud.height;
  irrad.resize(irrad.width * irrad.height);

  float squared_diff = 0.0f;

  for (size_t i = 0; i < index.size(); ++i)
  {
    size_t idx = index[i];

    Eigen::Vector3i color = cloud.points[idx].getRGBVector3i();
    Eigen::Vector3f colorf(color.x() / 255.0f, color.y() / 255.0f, color.z() / 255.0f);

    float denom = Al(i);

    // compute albedo
    Eigen::Vector3f albedof(colorf.x() / denom, colorf.y() / denom, colorf.z() / denom);
    //log_file << "albedo f " << albedof.transpose() << std::endl;

    // clamping
    albedof.x() = std::max(std::min(albedof.x(), 1.0f), 0.0f);
    albedof.y() = std::max(std::min(albedof.y(), 1.0f), 0.0f);
    albedof.z() = std::max(std::min(albedof.z(), 1.0f), 0.0f);

    // compute irradiance
    Eigen::Vector3f irradf(albedof.x() * denom, albedof.y() * denom, albedof.z() * denom);
    //log_file << "irrad f " << irradf.transpose() << std::endl;

    // clamping
    irradf.x() = std::max(std::min(irradf.x(), 1.0f), 0.0f);
    irradf.y() = std::max(std::min(irradf.y(), 1.0f), 0.0f);
    irradf.z() = std::max(std::min(irradf.z(), 1.0f), 0.0f);

    Eigen::Vector3i albedoi(albedof.x() * 255.0f, albedof.y() * 255.0f, albedof.z() * 255.0f);
    //log_file << "albedo i " << albedoi.transpose() << std::endl;

    albedo.points[idx].r = albedoi.x();
    albedo.points[idx].g = albedoi.y();
    albedo.points[idx].b = albedoi.z();

    Eigen::Vector3i irradi(irradf.x() * 255.0f, irradf.y() * 255.0f, irradf.z() * 255.0f);
    //log_file << "irrad i " << irradi.transpose() << std::endl;

    irrad.points[idx].r = irradi.x();
    irrad.points[idx].g = irradi.y();
    irrad.points[idx].b = irradi.z();

    Eigen::Vector3f diff = colorf - irradf;
    squared_diff += diff.dot(diff);

#ifdef _DEBUG
    /*log_file << "Al " << denom << std::endl;
    log_file << "color " << colorf.transpose() << std::endl;
    log_file << "albedo " << albedof.transpose() << std::endl;
    log_file << "irrad " << irradf.transpose() << std::endl;*/
    //log_file << "albedo " << albedo.points[idx].getRGBVector3i().transpose() << std::endl;
    //log_file << "irrad " << irrad.points[idx].getRGBVector3i().transpose() << std::endl;
#endif // _DEBUG
  }

#ifdef _DEBUG
  float mean_squared_diff = squared_diff / index.size();
  std::cout << "image - irradiance error: " << mean_squared_diff << std::endl;
#endif // _DEBUG
}
