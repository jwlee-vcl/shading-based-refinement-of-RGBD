#include "LightEstimator.h"

#include "SphericalHarmonics.h"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

void LightingEstimator::compute(
  const cv::Mat& depth, 
  const cv::Mat& grayf, 
  std::vector<float>& illum_coeffs, 
  cv::Mat& albedo_grayf,
  const cv::Point2f& focal,
  const cv::Point2f& pp)
{
  size_t width = depth.cols;
  size_t height = depth.rows;

  cv::Point2f inv_focal(1/focal.x, 1/focal.y);

  std::cout << "estimate normals" << std::endl;
  cv::Mat normals;
  cv::Mat points;
  compute_normals(depth, points, normals, inv_focal, pp);

  std::cout << "filter out pixels by angle" << std::endl;
  std::vector<size_t> index;
  filter_pixels_by_normal(points, normals, index);

  std::cout << "compute illumination coefficients" << std::endl;    
  compute_illum_coeffs(grayf, normals, index, illum_coeffs, albedo_grayf);
}

void LightingEstimator::compute_normals(
  const cv::Mat& depth,
  cv::Mat& points,
  cv::Mat& normals,
  const cv::Point2f& inv_focal,
  const cv::Point2f& pp)
{
  int width   = depth.cols;
  int height  = depth.rows;
   
  //std::cout << "depth 2 points" << std::endl;
  depth2point(depth, points, inv_focal, pp);

  normals.create(height, width, CV_32FC3);

  static cv::Vec3f nan(
    std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::quiet_NaN(), 
    std::numeric_limits<float>::quiet_NaN());

  //std::cout << "compute normals" << std::endl;
  for (int ir = 1; ir < depth.rows; ++ir)
  {
    for (int ic = 1; ic < depth.cols; ++ic)
    {
      if (depth.at<float>(ir, ic) == 0 ||
        depth.at<float>(ir, ic - 1) == 0 ||
        depth.at<float>(ir - 1, ic) == 0)
      {
        normals.at<cv::Vec3f>(ir, ic) = nan;        
        continue;
      }
      
      cv::Vec3f dv = points.at<cv::Vec3f>(ir - 1, ic) - points.at<cv::Vec3f>(ir, ic);
      cv::Vec3f dh = points.at<cv::Vec3f>(ir, ic - 1) - points.at<cv::Vec3f>(ir, ic);

      cv::Vec3f n = dv.cross(dh);

      normals.at<cv::Vec3f>(ir, ic) = cv::normalize(n);
    }
  }

  if (false)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);    
    cloud->width = width;
    cloud->height= height;
    cloud->points.resize(cloud->width*cloud->height);

    pcl::PointCloud<pcl::Normal>::Ptr normal(new pcl::PointCloud<pcl::Normal>);
    normal->width = width;
    normal->height = height;
    normal->points.resize(normal->width*normal->height);
    
    // fill cloud
    for (int ir = 1; ir < depth.rows; ++ir)
    {
      for (int ic = 1; ic < depth.cols; ++ic)
      {
        if (depth.at<float>(ir, ic) == 0 ||
          depth.at<float>(ir, ic - 1) == 0 ||
          depth.at<float>(ir - 1, ic) == 0)
        {
          (*cloud)(ic, ir).x = std::numeric_limits<float>::quiet_NaN();
          (*cloud)(ic, ir).y = std::numeric_limits<float>::quiet_NaN();
          (*cloud)(ic, ir).z = std::numeric_limits<float>::quiet_NaN();
          (*normal)(ic, ir).normal_x = std::numeric_limits<float>::quiet_NaN();
          (*normal)(ic, ir).normal_y = std::numeric_limits<float>::quiet_NaN();
          (*normal)(ic, ir).normal_z = std::numeric_limits<float>::quiet_NaN();
          continue;
        }

        (*cloud)(ic, ir).x = points.at<cv::Vec3f>(ir, ic)[0];
        (*cloud)(ic, ir).y = points.at<cv::Vec3f>(ir, ic)[1];
        (*cloud)(ic, ir).z = points.at<cv::Vec3f>(ir, ic)[2];

        (*normal)(ic, ir).normal_x = normals.at<cv::Vec3f>(ir, ic)[0];
        (*normal)(ic, ir).normal_y = normals.at<cv::Vec3f>(ir, ic)[1];
        (*normal)(ic, ir).normal_z = normals.at<cv::Vec3f>(ir, ic)[2];          
      }
    }

    pcl::visualization::PCLVisualizer viewer("normal_viewer");
    viewer.addPointCloud<pcl::PointXYZ>(cloud, "points");
    viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normal, 100, 0.1f, "normals");
    viewer.setSize(640, 640);

    viewer.spin();
  }
}

void LightingEstimator::filter_pixels_by_normal(
  const cv::Mat& points, 
  const cv::Mat& normals, 
  std::vector<size_t>& index)
{
  int width = points.cols;
  int height = points.rows;

  index.resize(width * height);
  validmap_ = cv::Mat::zeros(height, width, CV_8U);

  size_t j = 0;

  const static float angle_thresh = std::cosf(78.0f * M_PI / 180.0f);
  std::cout << "angle threshold " << angle_thresh << std::endl;

  for (size_t ir = 0; ir < height; ++ir)
  {
    for (size_t ic = 0; ic < width; ++ic)
    {
      // check normal
      if (!pcl_isfinite(normals.at<cv::Vec3f>(ir, ic)[0]) ||
        !pcl_isfinite(normals.at<cv::Vec3f>(ir, ic)[1]) ||
        !pcl_isfinite(normals.at<cv::Vec3f>(ir, ic)[2]))
      {
        continue;
      }

      // compare angle
      cv::Vec3f view_dir = -points.at<cv::Vec3f>(ir,ic);      
      view_dir = cv::normalize(view_dir);

      cv::Vec3f normal = normals.at<cv::Vec3f>(ir, ic);

      float angle = view_dir.dot(normal);

      if (angle > angle_thresh)
      {
        index[j] = (ir * width + ic);
        validmap_.at<unsigned char>(ir * width + ic) = 255;
        ++j;
      }
    }
  }
  index.resize(j);

  std::cout << "number of valid pixels: " << index.size() << std::endl;
}

void LightingEstimator::compute_illum_coeffs(
  const cv::Mat& grayf, 
  const cv::Mat& normals, 
  const std::vector<size_t>& index, 
  std::vector<float>& illum_coeffs, 
  cv::Mat& albedof)
{
  //std::cout << "build linear system" << std::endl;

  int width = grayf.cols;
  int height = grayf.rows;

  // build matrix
  // A l = I
  Eigen::MatrixXf A(index.size(), 9);
  Eigen::MatrixXf I(index.size(), 1);

  for (size_t i = 0; i < index.size(); ++i)
  {
    int r = index[i] / width;
    int c = index[i] % width;

    cv::Vec3f n = normals.at<cv::Vec3f>(r,c);
    
    std::vector<float> sh;
    SphericalHarmonics::eval_3_band(n, sh);

    A(i, 0) = sh[0];
    A(i, 1) = sh[1];
    A(i, 2) = sh[2];
    A(i, 3) = sh[3];
    A(i, 4) = sh[4];
    A(i, 5) = sh[5];
    A(i, 6) = sh[6];
    A(i, 7) = sh[7];
    A(i, 8) = sh[8];

    I(i, 0) = grayf.at<float>(r,c);
  }

  Eigen::MatrixXf AtA = A.transpose() * A;
  Eigen::MatrixXf AtI = A.transpose() * I;

  std::cout << "solve linear system" << std::endl;

  Eigen::MatrixXf l = AtA.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(AtI);

  Eigen::MatrixXf Al = A * l;

  std::cout << "illum coeffs" << l.transpose() << std::endl;

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
  
  albedof.create(height, width, CV_32F);
    
  // compute albedo image
  std::cout << "compute albedo image" << std::endl;
  
  for (size_t i = 0; i < index.size(); ++i)
  {
    size_t idx = index[i];

    float denom = Al(i);

    // compute albedo
    albedof.at<float>(idx) = grayf.at<float>(idx) / denom;
  }

  // clamping
  cv::max(albedof, 0.0f, albedof);
  cv::min(albedof, 1.0f, albedof);
  
  if(true)
  {    
    radf_.create(height, width, CV_32F);
    irradf_.create(height, width, CV_32F);

    for (size_t i = 0; i < index.size(); ++i)
    {
      size_t idx = index[i];

      float denom = Al(i);
      
      // build radiance
      radf_.at<float>(idx) = denom;

      // build irradiance
      irradf_.at<float>(idx) = albedof.at<float>(idx) * denom;
    }
    
    cv::Mat irradf(height, width, CV_32F);
    
    cv::imshow("grayf", grayf);
    cv::imshow("radiancef", radf_);
    cv::imshow("irradiancef", irradf_);
    cv::imshow("albedof", albedof);

    cv::Mat normalmap;
    compute_sphere_normal(normalmap, 555);

    cv::Mat lightmapf;
    compute_lightmap(lightmapf, normalmap, illum_coeffs);

    cv::max(lightmapf, 0.0f, lightmapf);
    cv::min(lightmapf, 1.0f, lightmapf);
    
    cv::imshow("lightmap", lightmapf);

    cv::waitKey();
    cv::destroyAllWindows();
  }
}

void LightingEstimator::depth2point(
  const cv::Mat& depth, 
  cv::Mat& points, 
  const cv::Point2f& inv_focal,
  const cv::Point2f& pp)
{
  int width = depth.cols;
  int height = depth.rows;

  points.create(height, width, CV_32FC3);

  static cv::Vec3f nan(
    std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::quiet_NaN());

  for (size_t ir = 0; ir < points.rows; ++ir)
  {
    for (size_t ic = 0; ic < points.cols; ++ic)
    {
      { // depth
        float val = depth.at<float>(ir, ic);
        if (0 == val)
        {
          points.at<cv::Vec3f>(ir, ic) = nan;
          continue;
        }

        float u = ic - pp.x;
        float v = ir - pp.y;

        float real_z = val;
        float real_x = u * real_z * inv_focal.x;
        float real_y = v * real_z * inv_focal.y;

        points.at<cv::Vec3f>(ir, ic)[0] = real_x;
        points.at<cv::Vec3f>(ir, ic)[1] = real_y;
        points.at<cv::Vec3f>(ir, ic)[2] = real_z;
      }
    }
  }
}

void LightingEstimator::compute_sphere_normal(
  cv::Mat& normals, int dim)
{
  if ((dim % 2) == 0)
  {
    dim += 1;
  }
  
  normals.create(dim, dim, CV_32FC3);
  
  for (int ir = 0; ir < dim; ++ir) 
  {
    for (int ic = 0; ic < dim; ++ic)
    {
      cv::Vec3f normal;
      normal[0] = 2.0f * (float)(ic - dim / 2) / (float)dim;
      normal[1] = 2.0f * (float)(ir - dim / 2) / (float)dim;

      float norm = (normal[0] * normal[0] + normal[1] * normal[1]);
      
      if (norm > 1.0f)
        continue;

      normal[2] = std::sqrt(1.0f - norm);

      normals.at<cv::Vec3f>(ir, ic) = normal;
    }
  }
}

void LightingEstimator::compute_lightmap(
  cv::Mat& lightmap, 
  const cv::Mat& normals, 
  const std::vector<float>& l)
{
  int cols = normals.cols;
  int rows = normals.rows;

  lightmap.create(rows, cols, CV_32F);

  std::vector<float> sh;
  sh.resize(9);

  for (int ir = 0; ir < rows; ++ir)
  {
    for (int ic = 0; ic < cols; ++ic)
    {
      cv::Vec3f normal = normals.at<cv::Vec3f>(ir, ic);
           
      SphericalHarmonics::eval_3_band(normal, sh);

      float val = l[0] * sh[0] + l[1] * sh[1] + l[2] * sh[2] +
        l[3] * sh[3] + l[4] * sh[4] + l[5] * sh[5] +
        l[6] * sh[6] + l[7] * sh[7] + l[8] * sh[8];

      lightmap.at<float>(ir, ic) = val;
    }
  }  
}
