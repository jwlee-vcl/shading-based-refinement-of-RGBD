#include "configs.h"

#include <pcl/pcl_macros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/console/print.h>

#include <pcl/io/pcd_io.h>
#include <pcl/io/openni2_grabber.h>
#include <vector>

#include <iostream>

using namespace std;
using namespace pcl;
using namespace pcl::console;

typedef PointCloud<PointXYZRGBA> Cloud;
typedef Cloud::ConstPtr CloudConstPtr;

int i = 0;

//////////////////////////////////////////////////////////////////////////////
void
printHelp(int, char **argv)
{
  //print_error("Syntax is: %s input.oni\n", argv[0]);
}

//////////////////////////////////////////////////////////////////////////////
void
cloud_cb(const CloudConstPtr& cloud)
{
  std::stringstream buf;
  buf << "frame_" << std::setfill('0') << std::setw(6) <<  i << ".pcd";
  pcl::io::savePCDFileBinary(buf.str(), *cloud);
  
  /*PCL_INFO("Wrote a cloud with %lu (%ux%u) points in %s.\n",
    cloud->size(), cloud->width, cloud->height, buf);*/

  ++i;
}

/* ---[ */
int
main(int argc, char **argv)
{
  //print_info("Convert an ONI file to PCD format. For more information, use: %s -h\n", argv[0]);

  if (argc < 2)
  {
    printHelp(argc, argv);
    return (-1);
  }

  pcl::io::OpenNI2Grabber::Mode depth_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;
  pcl::io::OpenNI2Grabber::Mode image_mode = pcl::io::OpenNI2Grabber::OpenNI_Default_Mode;

  pcl::Grabber* grabber = new pcl::io::OpenNI2Grabber(argv[1], depth_mode, image_mode);
  
  boost::function<void(const CloudConstPtr&) > f = boost::bind(&cloud_cb, _1);
  boost::signals2::connection c = grabber->registerCallback(f);

  grabber->start();

  while (grabber->isRunning())
  {    
  }

  grabber->stop();

  /*PCL_INFO("Successfully processed %d frames.\n", i);*/

  delete grabber;

  return (0);
}