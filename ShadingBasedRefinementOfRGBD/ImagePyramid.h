#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

class ImagePyramid
{
public:
  ImagePyramid(size_t num_levels = 3)
  {
    set_num_levels(num_levels);
  } 
  
  virtual void build_pyramid(const cv::Mat& imagef);
  
  virtual void prolongation(size_t target_level);

  inline void set_num_levels(int num_levels)
  {
    images_.resize(num_levels);
  }

  inline size_t get_num_levels() const
  {
    return images_.size();
  }

  inline cv::Mat& get_image(size_t level = 0)
  { 
    return images_[level];
  }

  inline const cv::Mat& get_image(size_t level = 0) const
  {
    return images_[level];
  }

  ImagePyramid& operator=(const ImagePyramid& other);
  
  ImagePyramid& operator=(ImagePyramid& other);  

protected:
  std::vector<cv::Mat> images_;  
};

class DepthPyramid: public ImagePyramid
{
public:
  DepthPyramid(size_t num_levels = 3)
    :ImagePyramid(num_levels)
  {}

  virtual void build_pyramid(const cv::Mat& imagef);

  virtual void prolongation(size_t target_level);

protected:
  void resize(const cv::Mat& src, cv::Mat& dst, cv::Size dsize);
};

class ValidPyramid : public ImagePyramid
{
public:
  ValidPyramid(size_t num_levels = 3)
    :ImagePyramid(num_levels)
  {}

  virtual void build_pyramid(const cv::Mat& imagef);  

  // no use
  //virtual void prolongation(size_t target_level);
};


