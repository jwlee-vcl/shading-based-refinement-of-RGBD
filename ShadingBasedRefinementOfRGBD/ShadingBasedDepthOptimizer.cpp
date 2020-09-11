#include "ShadingBasedDepthOptimizer.h"

#include <pcl/surface/organized_fast_mesh.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/io/ply_io.h>

ShadingBasedDepthOptimizer::ShadingBasedDepthOptimizer(
  OptimizerSettings& settings)
  :settings_(settings)
{
}

ShadingBasedDepthOptimizer::~ShadingBasedDepthOptimizer()
{

}

void ShadingBasedDepthOptimizer::set_data(
  const cv::Mat& grayf, const cv::Mat& depthf)
{
  if (settings_.apply_blur)
  {
    cv::bilateralFilter(depthf, initial_depthf_,
      settings_.blur_size, settings_.sigma_color, settings_.sigma_space);    
  }
  else
  {
    initial_depthf_ = depthf.clone();
  }
  
  grayf_ = grayf.clone();  
}

void ShadingBasedDepthOptimizer::compute()
{
  std::vector<size_t> valid_pixels;
  std::vector<float> illum_coeffs;
  cv::Mat albedof;

  std::cout << "Estimate illumination" << std::endl;

  LightingEstimator le;
  le.compute(initial_depthf_, grayf_, 
    illum_coeffs, albedof, settings_.focal, settings_.pp);  

  const cv::Mat& validmap = le.get_validmap();
  
  if (settings_.use_multigrid)
  {
    depth_optimize_multigrid(validmap, illum_coeffs, albedof);
  }
  else
  {    
    depth_optimize(validmap, illum_coeffs, albedof);
  }
}

void ShadingBasedDepthOptimizer::depth_optimize(
  const cv::Mat& validmap, 
  const std::vector<float>& illum_coeffs, cv::Mat& albedof)
{
  std::cout << "Depth optimization" << std::endl;

  DepthOptimizer optimizer;

  optimizer.set_weights(settings_.w_grad, settings_.w_smooth, settings_.w_depth);

  optimizer.compute(initial_depthf_, refined_depthf_,
    validmap,
    grayf_, albedof, illum_coeffs,
    settings_.focal, settings_.pp, settings_.num_iter);
}

void ShadingBasedDepthOptimizer::depth_optimize_multigrid(
  const cv::Mat& validmap,
  const std::vector<float>& illum_coeffs, cv::Mat& albedof)
{
  ImagePyramid images_grayf(settings_.num_scale);
  images_grayf.build_pyramid(grayf_);

  ImagePyramid albedos_grayf(settings_.num_scale);  
  albedos_grayf.build_pyramid(albedof);

  DepthPyramid initial_depthf(settings_.num_scale);
  initial_depthf.build_pyramid(initial_depthf_);
  
  DepthPyramid refined_depthf(settings_.num_scale);
  refined_depthf = initial_depthf;
  
  ValidPyramid validmaps(settings_.num_scale);
  validmaps.build_pyramid(validmap);

  DepthOptimizer optimizer;

  optimizer.set_weights(settings_.w_grad, settings_.w_smooth, settings_.w_depth);

  int num_levels = images_grayf.get_num_levels();

  const cv::Point2f& focal = settings_.focal;
  const cv::Point2f& pp = settings_.pp;

  for (int lev = (num_levels-1); lev > -1; --lev)
  {
    std::cout << "lev - " << lev << std::endl;

    float factor = std::powf(2.0f, lev);

    cv::Point2f lev_focal(focal.x / factor, focal.y / factor);
    cv::Point2f lev_pp(pp.x / factor, pp.y / factor);

    if (lev == (num_levels - 1))
    {
      std::cout << "initial optimizer" << lev << std::endl;

      refined_depthf.get_image(lev) = initial_depthf.get_image(lev).clone();

      optimizer.compute(initial_depthf.get_image(lev), refined_depthf.get_image(lev),
        validmaps.get_image(lev),
        images_grayf.get_image(lev), albedos_grayf.get_image(lev),
        illum_coeffs,
        lev_focal, lev_pp, settings_.num_iter);
    }
    else
    {
      std::cout << "next level" << lev << std::endl;

      //cv::Mat init_depth = refined_depthf.get_image(lev).clone();
      
      optimizer.compute(initial_depthf.get_image(lev), refined_depthf.get_image(lev),
      //optimizer.compute(init_depth, refined_depthf.get_image(lev),
        validmaps.get_image(lev),
        images_grayf.get_image(lev), albedos_grayf.get_image(lev),
        illum_coeffs,
        lev_focal, lev_pp, settings_.num_iter);
    }

    if (lev > 0)
    {
      std::cout << "prolongation" << std::endl;

      refined_depthf.prolongation(lev - 1);
    }    

    //visualize_depth(refined_depthf.get_image(lev), lev_focal, lev_pp, "refined_depth");
  }
  
  refined_depthf_ = refined_depthf.get_image(0).clone();
}

void ShadingBasedDepthOptimizer::visualize_depth(
  const cv::Mat& depthf, 
  const cv::Point2f& focal, const cv::Point2f& pp, 
  const std::string& name /*= "depth"*/)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  depth2point(depthf, *cloud, focal, pp);

  pcl::PolygonMesh mesh;
  
  pcl::OrganizedFastMesh<pcl::PointXYZ> ofm;
  ofm.setInputCloud(cloud);
  ofm.reconstruct(mesh);

  //pcl::io::savePLYFile(name+"_mesh.ply", mesh);

  pcl::visualization::PCLVisualizer pcl_viewer(name);
  pcl_viewer.setSize(640, 640);
  pcl_viewer.setBackgroundColor(0, 0, 0);
  pcl_viewer.addPolygonMesh(mesh);
  pcl_viewer.setCameraPosition(
    0.0f, 0.0f, -0.3f, // eye
    0.0f, 0.0f, 1.0f, // center
    0.0f, -1.0f, 0.0f); // up  

  pcl_viewer.spin();
}

void ShadingBasedDepthOptimizer::depth2point(
  const cv::Mat& depthf, 
  pcl::PointCloud<pcl::PointXYZ>& cloud, 
  const cv::Point2f& focal, const cv::Point2f& pp)
{
  int width = depthf.cols;
  int height = depthf.rows;

  cloud.width = width;
  cloud.height = height;
  cloud.resize(cloud.width * cloud.height);
  cloud.is_dense = false;

  cv::Point2f inverse_focal = cv::Point2f(1.0f / focal.x, 1.0f / focal.y);

  for (size_t ir = 0; ir < depthf.rows; ++ir)
  {
    for (size_t ic = 0; ic < depthf.cols; ++ic)
    {
      { // depth
        float val = depthf.at<float>(ir, ic);
        if (0 == val)
        {
          cloud(ic, ir).x = std::numeric_limits<float>::quiet_NaN();
          cloud(ic, ir).y = std::numeric_limits<float>::quiet_NaN();
          cloud(ic, ir).z = std::numeric_limits<float>::quiet_NaN();

          continue;
        }

        float u = ic - pp.x;
        float v = ir - pp.y;

        float real_z = val;
        float real_x = u * real_z * inverse_focal.x;
        float real_y = v * real_z * inverse_focal.y;

        cloud(ic, ir).x = real_x;
        cloud(ic, ir).y = real_y;
        cloud(ic, ir).z = real_z;
      }
    }
  }
}
