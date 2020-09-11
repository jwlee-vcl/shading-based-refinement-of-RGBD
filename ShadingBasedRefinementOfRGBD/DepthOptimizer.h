#pragma once

#include <opencv2/opencv.hpp>

#include <Eigen/Sparse>

#include <ctime>
#include <fstream>

class DepthOptimizer
{
public:
  DepthOptimizer()
    :w_grad_(1.0f), w_smooth_(100.0f), w_depth_(10.0f)
  { 
#ifdef _DEBUG
    const std::time_t curr_time = std::time(0);
    char timestr[128];

    std::strftime(timestr, 100, "%Y-%m-%d-%H-%M-%S", std::localtime(&curr_time));

    log_file_.open(std::string(timestr) + "_DepthOptimizer.txt");

    if (!log_file_.is_open())
    {
      std::cout << timestr << "failed to create a log file" << std::endl;
    }
#endif
  }

  ~DepthOptimizer()
  {
#ifdef _DEBUG
    if (!log_file_.is_open())
    {
      log_file_.close();
    }
#endif
  }
  
  void compute(
    const cv::Mat& initial_depthf,
    cv::Mat& depthf,
    const cv::Mat& validmap,
    const cv::Mat& grayf,
    const cv::Mat& albedo_grayf,
    const std::vector<float>& illum_coeffs,
    const cv::Point2f& focal, const cv::Point2f& pp,
    const size_t num_iter = 2);

  void set_weights(
    float w_grad = 1.0f,
    float w_smooth = 100.0f,
    float w_depth = 10.0f)
  {
    w_grad_ = w_grad;
    w_smooth_ = w_smooth;
    w_depth_ = w_depth;
  }

  void compute_irradiance(
    cv::Mat& irradf,
    const cv::Mat& depthf,    
    const std::vector<size_t>& valid_pixels,
    const cv::Mat& albedo_grayf,
    const std::vector<float>& illum_coeffs,
    const cv::Point2f& inv_focal, const cv::Point2f& pp, bool clamp_val = false);

  void compute_radiance(
    cv::Mat& radf,
    const cv::Mat& depthf,
    const std::vector<size_t>& valid_pixels,
    const std::vector<float>& illum_coeffs,  
    const cv::Point2f& inv_focal, const cv::Point2f& pp);

  void filter_valid_pixels(
    const cv::Mat& depthf, const cv::Mat& validmap,
    std::vector<size_t>& valid_pixels);

  void test_derivative();

protected:  
  cv::Vec3f compute_normal(
    const cv::Mat& depthf,
    const size_t i, const size_t j,
    const cv::Point2f& inv_focal, const cv::Point2f& pp);

  cv::Vec3f compute_normal(
    const float Dij, const float Dl, const float Du,
    const size_t i, const size_t j,
    const cv::Point2f& inv_focal, const cv::Point2f& pp);

  cv::Vec3f compute_normal_unnormalized(
    const float Dij, const float Dl, const float Du,
    const size_t i, const size_t j,
    const cv::Point2f& inv_focal, const cv::Point2f& pp);

  void compute_normals(
    const cv::Mat& depthf, cv::Mat& normals,
    const cv::Point2f& inv_focal, const cv::Point2f& pp);

  // Energy terms
  float compute_B(const cv::Vec3f& n, 
    const std::vector<float>& l, const float k);

  float compute_r1(
    const float Dij, const float Dl, const float Du, const float Dr, const float Dru,
    const size_t i, const size_t j,
    const cv::Point2f& inv_focal, const cv::Point2f& pp,
    const std::vector<float>& l, 
    const float kij, const float kr, 
    const float Iij, const float Ir);

  float compute_r2(
    const float Dij, const float Dl, const float Du, const float Dd, const float Ddl,
    const size_t i, const size_t j,
    const cv::Point2f& inv_focal, const cv::Point2f& pp,
    const std::vector<float>& l,
    const float kij, const float kd,
    const float Iij, const float Id);

  float compute_r3(
    const float Dij, const float Dl, const float Dr, const float Du, const float Dd,
    const size_t i, const size_t j,
    const cv::Point2f& inv_focal, const cv::Point2f& pp, float ws = 0.25f);

  float compute_r4(
    const float Dij, const float Dl, const float Dr, const float Du, const float Dd,
    const size_t i, const size_t j,
    const cv::Point2f& inv_focal, const cv::Point2f& pp, float ws = 0.25f);

  float compute_r5(
    const float Dij, const float Dl, const float Dr, const float Du, const float Dd,
    const size_t i, const size_t j,
    const cv::Point2f& inv_focal, const cv::Point2f& pp, float ws = 0.25f);

  float compute_r6(const float Dij, const float initial_Dij);

  // Elements of Jacobian
  cv::Vec3f compute_dBdn(
    const cv::Vec3f& n,
    const std::vector<float>& l,
    const float k);

  std::vector<cv::Vec3f> compute_dndn_til(
    const cv::Vec3f& n, const cv::Vec3f& n_hat);

  cv::Vec3f compute_dn_tildDij(
    const float Dij, const float Dl, const float Du,
    const size_t i, const size_t j,
    const cv::Point2f& inv_focal, const cv::Point2f& pp);

  cv::Vec3f compute_dn_tildDl(
    const float Dij, const float Dl, const float Du,
    const size_t i, const size_t j,
    const cv::Point2f& inv_focal, const cv::Point2f& pp);

  cv::Vec3f compute_dn_tildDu(
    const float Dij, const float Dl, const float Du,
    const size_t i, const size_t j,
    const cv::Point2f& inv_focal, const cv::Point2f& pp);
 
  float compute_dr1dDij( // D(i,j)
    const float Dij,
    const float Dl, const float Du, const float Dr, const float Dru,
    const size_t i, const size_t j,
    const std::vector<float>& l, 
    const float kij, const float kr,
    const cv::Point2f& inv_focal, const cv::Point2f& pp);

  float compute_dr1dDl( // D(i-1,j)
    const float Dij, const float Dl, const float Du,
    const size_t i, const size_t j,
    const std::vector<float>& l, const float kij,
    const cv::Point2f& inv_focal, const cv::Point2f& pp);

  float compute_dr1dDu( // D(i,j-1)
    const float Dij, const float Dl, const float Du,
    const size_t i, const size_t j,
    const std::vector<float>& l, const float kij,
    const cv::Point2f& inv_focal, const cv::Point2f& pp);

  float compute_dr1dDr( // D(i+1,j)
    const float Dij, const float Dr, const float Dru,
    const size_t i, const size_t j,
    const std::vector<float>& l, const float kr,
    const cv::Point2f& inv_focal, const cv::Point2f& pp);

  float compute_dr1dDru( // D(i+1,j-1)
    const float Dij, const float Dr, const float Dru,
    const size_t i, const size_t j,
    const std::vector<float>& l, const float kr,
    const cv::Point2f& inv_focal, const cv::Point2f& pp);

  float compute_dr2dDij( // D(i,j)
    const float Dij,
    const float Dl, const float Du, const float Dd, const float Ddl,
    const size_t i, const size_t j,
    const std::vector<float>& l, const float kij, const float kd,
    const cv::Point2f& inv_focal, const cv::Point2f& pp);

  float compute_dr2dDl( // D(i-1,j)
    const float Dij, const float Dl, const float Du,
    const size_t i, const size_t j,
    const std::vector<float>& l, const float kij,
    const cv::Point2f& inv_focal, const cv::Point2f& pp);

  float compute_dr2dDu( // D(i,j-1)
    const float Dij, const float Dl, const float Du,
    const size_t i, const size_t j,
    const std::vector<float>& l, const float kij,
    const cv::Point2f& inv_focal, const cv::Point2f& pp);
    
  float compute_dr2dDd( // D(i,j+1)
    const float Dij, const float Dd, const float Ddl,
    const size_t i, const size_t j,
    const std::vector<float>& l, const float kd,
    const cv::Point2f& inv_focal, const cv::Point2f& pp);

  float compute_dr2dDdl( // D(i-1,j+1)
    const float Dij, const float Dd, const float Ddl,
    const size_t i, const size_t j,
    const std::vector<float>& l, const float kd,
    const cv::Point2f& inv_focal, const cv::Point2f& pp);

  float compute_dr3dDij( // D(i,j)
    const size_t i, const size_t j,
    const cv::Point2f& inv_focal, const cv::Point2f& pp);

  float compute_dr3dDl( // D(i-1,j)
    const size_t i, const size_t j,
    const cv::Point2f& inv_focal, const cv::Point2f& pp,
    const float ws = 0.25f);

  float compute_dr3dDu( // D(i,j-1)
    const size_t i, const size_t j,
    const cv::Point2f& inv_focal, const cv::Point2f& pp,
    const float ws = 0.25f);

  float compute_dr3dDr( // D(i+1,j)
    const size_t i, const size_t j,
    const cv::Point2f& inv_focal, const cv::Point2f& pp,
    const float ws = 0.25f);

  float compute_dr3dDd( // D(i,j+1)
    const size_t i, const size_t j,
    const cv::Point2f& inv_focal, const cv::Point2f& pp,
    const float ws = 0.25f);

  float compute_dr4dDij( // D(i,j)
    const size_t i, const size_t j,
    const cv::Point2f& inv_focal, const cv::Point2f& pp);

  float compute_dr4dDl( // D(i-1,j)
    const size_t i, const size_t j,
    const cv::Point2f& inv_focal, const cv::Point2f& pp,
    const float ws = 0.25f);

  float compute_dr4dDu( // D(i,j-1)
    const size_t i, const size_t j,
    const cv::Point2f& inv_focal, const cv::Point2f& pp,
    const float ws = 0.25f);

  float compute_dr4dDr( // D(i+1,j)
    const size_t i, const size_t j,
    const cv::Point2f& inv_focal, const cv::Point2f& pp,
    const float ws = 0.25f);

  float compute_dr4dDd( // D(i,j+1)
    const size_t i, const size_t j,
    const cv::Point2f& inv_focal, const cv::Point2f& pp,
    const float ws = 0.25f);

  float compute_dr5dDij(); // D(i,j)
  
  float compute_dr5dDl(const float ws = 0.25f); // D(i-1,j)
  
  float compute_dr5dDu(const float ws = 0.25f); // D(i,j-1)

  float compute_dr5dDr(const float ws = 0.25f); // D(i+1,j)
  
  float compute_dr5dDd(const float ws = 0.25f); // D(i,j+1)

  float compute_dr6dDij(); // D(i,j)

public:
  float w_grad_;
  float w_smooth_;
  float w_depth_;
    
  Eigen::VectorXf Fd;
  Eigen::VectorXf delta;

  Eigen::SparseMatrix<float> Jd;

  cv::Mat albedo_grayf_;

  std::ofstream log_file_;
};

