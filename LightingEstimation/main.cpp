#include "config.h"

#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingOpenGL);

#include <pcl/io/openni2_grabber.h>
#include <pcl/io/pcd_io.h>

#include "LightEstimator.h"

#include <pcl/range_image/range_image.h>

#ifdef _DEBUG
std::ofstream log_file;
#endif

//typedef pcl::PointXYZRGB PointT;
//typedef pcl::Normal NormalT;

void refine_depth(const pcl::RangeImage& src,
  pcl::RangeImage& dst,  
  const pcl::PointCloud<NormalT>& normals,
  const std::vector<size_t>& valid_pixels,
  const std::vector<float>& illum_coeffs,
  const pcl::PointCloud<pcl::RGB>& albedo,
  const pcl::PointCloud<pcl::RGB>& irrad)
{
  
}

int main()
{
#ifdef _DEBUG
  //log_file.open("log.txt");
#endif

  std::cout << "load point cloud" << std::endl;
  pcl::PointCloud<PointT> cloud;
  pcl::io::loadPCDFile("augustus_000000.pcd", cloud);
  
  std::cout << "size of cloud " << cloud.width << " X " << cloud.height << std::endl;

  pcl::PointCloud<NormalT> normals;
  std::vector<size_t> valid_pixels;
  std::vector<float> illum_coeffs;
  pcl::PointCloud<pcl::RGB> albedo;
  pcl::PointCloud<pcl::RGB> irrad;

  std::cout << "Estimate illumination" << std::endl;
  
  LightingEstimator le;
  le.compute(cloud, normals, valid_pixels, illum_coeffs, albedo, irrad);

  pcl::RangeImage depthmap;
  //depthmap.createFromPointCloud(cloud, );


#ifdef _DEBUG
  //log_file.close();
#endif
  
  return 0;
}