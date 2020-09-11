#include "ImagePyramid.h"

void ImagePyramid::build_pyramid(const cv::Mat& imagef)
{
  images_[0] = imagef;

  for (size_t i = 1; i < images_.size(); ++i)
  {
    cv::resize(images_[i - 1], images_[i], cv::Size(), 0.5f, 0.5f, cv::INTER_LINEAR);
  }
}

void ImagePyramid::prolongation(size_t target_level)
{
  const cv::Mat& src = images_[target_level + 1];
  cv::Mat& dst = images_[target_level];

  cv::resize(src, dst, cv::Size(), 2, 2, cv::INTER_CUBIC);
}

ImagePyramid& ImagePyramid::operator=(const ImagePyramid& other)
{
  size_t num_levels = other.get_num_levels();
  images_.resize(num_levels);

  for (size_t i = 0; i < num_levels; ++i)
  {
    images_[i] = other.get_image(i).clone();
  }

  return *this;
}

ImagePyramid& ImagePyramid::operator=(ImagePyramid& other)
{
  size_t num_levels = other.get_num_levels();
  images_.resize(num_levels);

  for (size_t i = 0; i < num_levels; ++i)
  {
    images_[i] = other.get_image(i).clone();
  }

  return *this;
}

void DepthPyramid::build_pyramid(const cv::Mat& imagef)
{
  int rows = imagef.rows;
  int cols = imagef.cols;

  images_[0] = imagef;

  for (size_t i = 1; i < images_.size(); ++i)
  {
    int width = cols >> i;
    int height = rows >> i;

    resize(images_[i - 1], images_[i], cv::Size(width, height));
    //cv::resize(images_[i - 1], images_[i], cv::Size(), 0.5f, 0.5f, cv::INTER_LINEAR);
  }
}

void DepthPyramid::prolongation(size_t target_level)
{
  const cv::Mat& src = images_[target_level + 1];
  cv::Mat& dst = images_[target_level];

  cv::resize(src, dst, cv::Size(), 2, 2, cv::INTER_CUBIC);

  /*
  cv::Size dsize((src.cols << 1), (src.rows << 1));
  dst.create(dsize, CV_32F);

  for (int ir = 0; ir < (src.rows - 1); ++ir)
  {
    for (int ic = 0; ic < (src.cols - 1); ++ic)
    {
      float val_0 = src.at<float>(ir, ic);
      float val_1 = src.at<float>(ir, ic + 1);
      float val_2 = src.at<float>(ir + 1, ic);
      float val_3 = src.at<float>(ir + 1, ic + 1);
      
      dst.at<float>(ir << 1, ic << 1) = val_0;

      float val_01 = val_0 + val_1;
      if (val_0 > 0 && val_1 > 0)
      {
        val_01 *= 0.5f;
      }        
      dst.at<float>(ir << 1, (ic << 1) + 1) = val_01;

      float val_02 = val_0 + val_2;
      if (val_0 > 0 && val_2 > 0)
      {
        val_02 *= 0.5f;
      }
      dst.at<float>((ir << 1) + 1, ic << 1) = val_02;
            
      float val_23 = val_2 + val_3;
      if (val_2 > 0 && val_3 > 0)
      {
        val_23 *= 0.5f;
      }
      
      float val = val_01 + val_23;
      if (val_01 > 0 && val_23 > 0)
      {
        val *= 0.5f;
      }

      dst.at<float>((ir << 1) + 1, (ic << 1) + 1) = val;
    }
  }  */
}

void DepthPyramid::resize(const cv::Mat& src, cv::Mat& dst, cv::Size dsize)
{
  if ((src.cols / dsize.width) != 2 || (src.rows / dsize.height) != 2)
  {
    std::cerr << "(src.cols / dsize.width) != 2 || (src.rows / dsize.height) != 2" << std::endl;
    return;
  }

  dst.create(dsize, CV_32F);

  for (int ir = 0; ir < dsize.height; ++ir)
  {
    for (int ic = 0; ic < dsize.width; ++ic)
    {
      float sum = 0;
      int count = 0;

      float val_0 = src.at<float>(ir << 1, ic << 1);
      float val_1 = src.at<float>((ir << 1), (ic << 1) + 1);
      float val_2 = src.at<float>((ir << 1) + 1, ic << 1);
      float val_3 = src.at<float>((ir << 1) + 1, (ic << 1) + 1);

      if (val_0 > 0) ++count;
      if (val_1 > 0) ++count;
      if (val_2 > 0) ++count;
      if (val_3 > 0) ++count;

      if (count > 0)
        sum = (val_0 + val_1 + val_2 + val_3) / count;
      else
        sum = 0;

      dst.at<float>(ir, ic) = sum;
    }
  }
}

void ValidPyramid::build_pyramid(const cv::Mat& imagef)
{
  int rows = imagef.rows;
  int cols = imagef.cols;

  images_[0] = imagef;

  for (size_t i = 1; i < images_.size(); ++i)
  {
    int width = cols >> i;
    int height = rows >> i;

    // TODO
    //cv::resize(images_[i - 1], images_[i], cv::Size(width, height), 0, 0, cv::INTER_NEAREST);

    images_[i].create(height, width, CV_8U);

    for (int ir = 0; ir < height; ++ir)
    {
      for (int ic = 0; ic < width; ++ic)
      {
        unsigned char val_0 = images_[i-1].at<unsigned char>(ir << 1, ic << 1);
        unsigned char val_1 = images_[i-1].at<unsigned char>((ir << 1), (ic << 1) + 1);
        unsigned char val_2 = images_[i-1].at<unsigned char>((ir << 1) + 1, ic << 1);
        unsigned char val_3 = images_[i-1].at<unsigned char>((ir << 1) + 1, (ic << 1) + 1);

        if (val_0 == 0 || val_1 == 0 || val_2 == 0 || val_3 == 0)
        {
          images_[i].at<unsigned char>(ir, ic) = 0;
        }          
        else
        {
          images_[i].at<unsigned char>(ir, ic) = 255;
        }          
      }
    }
  }
}
