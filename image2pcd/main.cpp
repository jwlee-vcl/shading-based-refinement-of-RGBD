#include "configs_pcl.h"
#include "configs_opencv.h"

#include <opencv2/opencv.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/io/pcd_io.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <limits>

int main()
{
  cv::Mat color = cv::imread("frame-000000.color.png", cv::IMREAD_UNCHANGED);
  cv::Mat depth = cv::imread("frame-000000.depth.png", cv::IMREAD_UNCHANGED);

  std::cout << "num color channels: " << color.channels() << std::endl;
  std::cout << "num color depth: " << color.depth() << std::endl;

  std::cout << "num depth channels: " << depth.channels() << std::endl;
  std::cout << "num depth depth: " << depth.depth() << std::endl;
  
  int width  = depth.cols;
  int height = depth.rows;

  cv::Mat color_resized;
  cv::resize(color, color_resized, cv::Size(width, height));

  //cv::imshow("color", color_resized);
  //cv::imshow("depth", depth);
  //cv::waitKey(0);

  float focal = 574.053f;
  float inverse_focal = 1.0/focal;
  float px = 320.0f;
  float py = 240.0f;
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  
  cloud->width = width;
  cloud->height = height;
  cloud->resize(cloud->width * cloud->height);
  
  for (size_t ir = 0; ir<cloud->height; ++ir)
  {
    for (size_t ic = 0; ic < cloud->width; ++ic)
    {
      {
        cv::Vec3b val = color_resized.at<cv::Vec3b>(ir, ic);

        (*cloud)(ic, ir).r = val[2];
        (*cloud)(ic, ir).g = val[1];
        (*cloud)(ic, ir).b = val[0];
      }
      {
        short val = depth.at<short>(ir, ic);
        if (0 == val)
        {
          (*cloud)(ic, ir).x = std::numeric_limits<float>::quiet_NaN();
          (*cloud)(ic, ir).y = std::numeric_limits<float>::quiet_NaN();
          (*cloud)(ic, ir).z = std::numeric_limits<float>::quiet_NaN();
          continue;
        }
        
        float u = ic - px;
        float v = ir - py;
        
        float real_z = val * 0.001f;
        float real_x = u * real_z * inverse_focal;
        float real_y = v * real_z * inverse_focal;
                
        (*cloud)(ic, ir).x = real_x;
        (*cloud)(ic, ir).y = real_y;
        (*cloud)(ic, ir).z = real_z;
      }      
    }
  }  

  /*pcl::visualization::PCLVisualizer viewer("cloud viewer");
  viewer.setBackgroundColor(0.0, 0.0, 0.0);
  viewer.addPointCloud<pcl::PointXYZRGB>(cloud, "cloud");
  viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");  
  viewer.addCoordinateSystem(0.1);
  viewer.initCameraParameters();

  pcl::visualization::ImageViewer image_viewer("image_viewer");
  image_viewer.addRGBImage(*cloud);

  while (!viewer.wasStopped() &&
    !image_viewer.wasStopped())
  {
    viewer.spinOnce();
    image_viewer.spinOnce();
  }*/
  
  pcl::io::savePCDFileBinary("augustus_000000.pcd", *cloud);  
  
  return 0;
}