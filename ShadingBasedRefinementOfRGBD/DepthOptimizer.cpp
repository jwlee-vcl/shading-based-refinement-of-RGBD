#include "DepthOptimizer.h"
#include "SphericalHarmonics.h"

void DepthOptimizer::compute(
  const cv::Mat& initial_depthf, cv::Mat& depthf,
  const cv::Mat& validmap,
  const cv::Mat& grayf, const cv::Mat& albedo_grayf,
  const std::vector<float>& illum_coeffs,
  const cv::Point2f& focal, const cv::Point2f& pp,
  const size_t num_iter)
{
  if (false)
  {
    cv::imshow("initial depth", initial_depthf);
    cv::imshow("validmap", validmap);
    cv::imshow("gray", grayf);
    cv::imshow("albedo gray", albedo_grayf);
    cv::waitKey(0);
  }

  size_t width  = (size_t)initial_depthf.cols;
  size_t height = (size_t)initial_depthf.rows;

  // filter valid pixels
  std::vector<size_t> indices;
  filter_valid_pixels(initial_depthf, validmap, indices);

  // for computing convenience
  std::vector<size_t> inverse_map;
  inverse_map.resize(width*height, -1);

  for (size_t i = 0; i < indices.size(); ++i)
  {
    // inverse map(i,j) store index of valid pixels
    inverse_map[indices[i]] = i;
  }

  // depth refine
  cv::Point2f inv_focal(1.0f / focal.x, 1.0f / focal.y);

  if (depthf.empty())
  {
    depthf = initial_depthf.clone();
  } 

  int num_eq = 6;
  int num_depth = indices.size();

  Eigen::SparseMatrix<float> Jd(num_eq * num_depth, num_depth);
  Eigen::VectorXf Fd(num_eq * num_depth);
  Eigen::VectorXf delta(num_depth);

  std::vector< Eigen::Triplet<float> > triplets;
  triplets.reserve(num_eq * num_depth * 7);

  cv::Mat points;

  float ws = 0.25f;

  float sqrt_w_grad = std::sqrt(w_grad_);
  float sqrt_w_smooth = std::sqrt(w_smooth_);
  float sqrt_w_depth = std::sqrt(w_depth_);

  albedo_grayf_ = albedo_grayf.clone();

  if(false)
  {
    cv::Mat irradf;
    compute_irradiance(irradf, depthf, indices, albedo_grayf_, illum_coeffs, inv_focal, pp, false);

    cv::imshow("initial irrad", irradf);   

    cv::waitKey();
    cv::destroyAllWindows();
  }  

  float err_grad = 0;
  float err_smooth = 0;
  float err_depth = 0;

  if(true)
  { 
    // compute initial error
    for (size_t k = 0; k < indices.size(); ++k)
    {
      int i = indices[k] % width;
      int j = indices[k] / width;

      float Dij = depthf.at<float>(j, i);
      float Dl = depthf.at<float>(j, i - 1);
      float Dr = depthf.at<float>(j, i + 1);
      float Du = depthf.at<float>(j - 1, i);
      float Dd = depthf.at<float>(j + 1, i);
      float Dru = depthf.at<float>(j - 1, i + 1);
      float Ddl = depthf.at<float>(j + 1, i - 1);

      float initial_Dij = initial_depthf.at<float>(j, i);

      float kij = albedo_grayf_.at<float>(j, i);
      float kr = albedo_grayf_.at<float>(j, i + 1);
      float kd = albedo_grayf_.at<float>(j + 1, i);

      float Iij = grayf.at<float>(j, i);
      float Ir = grayf.at<float>(j, i + 1);
      float Id = grayf.at<float>(j + 1, i);

      float r1 = compute_r1(Dij, Dl, Du, Dr, Dru, i, j, inv_focal, pp, illum_coeffs, kij, kr, Iij, Ir);
      float r2 = compute_r2(Dij, Dl, Du, Dd, Ddl, i, j, inv_focal, pp, illum_coeffs, kij, kd, Iij, Id);
      float r3 = compute_r3(Dij, Dl, Dr, Du, Dd, i, j, inv_focal, pp, ws);
      float r4 = compute_r4(Dij, Dl, Dr, Du, Dd, i, j, inv_focal, pp, ws);
      float r5 = compute_r5(Dij, Dl, Dr, Du, Dd, i, j, inv_focal, pp, ws);
      float r6 = compute_r6(Dij, initial_Dij);

      err_grad += (r1*r1 + r2*r2);
      err_smooth += (r3*r3 + r4*r4 + r5*r5);
      err_depth += (r6*r6);
    }

    std::cout << "Initial error-";
    std::cout << " E_grad(d): " << err_grad;
    std::cout << ", E_smooth(d): " << err_smooth;
    std::cout << ", E_depth(d): " << err_depth << std::endl;
  }

  // step
  for (size_t iter = 0; iter < num_iter; ++iter)
  {
    std::cout << iter << " iter" << std::endl;

    triplets.clear();
    
    // build matrix
    for (size_t k = 0; k < indices.size(); ++k)
    {
      if ((k % 10000) == 0)
      {
        std::cout << k << " th depth" << std::endl;
      }
      
      size_t i = indices[k] % width;
      size_t j = indices[k] / width;

      int left = inverse_map[j*width + (i - 1)];
      int right = inverse_map[j*width + (i + 1)];
      int up = inverse_map[(j - 1)*width + i];
      int down = inverse_map[(j + 1)*width + i];
      int leftdown = inverse_map[(j + 1)*width + (i - 1)];
      int rightup = inverse_map[(j - 1)*width + (i + 1)];

      float Dij = depthf.at<float>(j, i);
      float Dl  = depthf.at<float>(j, i - 1);
      float Dr  = depthf.at<float>(j, i + 1);
      float Du  = depthf.at<float>(j - 1, i);
      float Dd  = depthf.at<float>(j + 1, i);
      float Dru = depthf.at<float>(j - 1, i + 1);
      float Ddl = depthf.at<float>(j + 1, i - 1);
      
      float initial_Dij = initial_depthf.at<float>(j, i);

      float kij = albedo_grayf_.at<float>(j, i);
      float kr  = albedo_grayf_.at<float>(j, i + 1);
      float kd  = albedo_grayf_.at<float>(j + 1, i);

      float Iij = grayf.at<float>(j, i);
      float Ir  = grayf.at<float>(j, i + 1);
      float Id  = grayf.at<float>(j + 1, i);

      { // compute error
        float r1 = compute_r1(Dij, Dl, Du, Dr, Dru, i, j, inv_focal, pp, illum_coeffs, kij, kr, Iij, Ir);
        float r2 = compute_r2(Dij, Dl, Du, Dd, Ddl, i, j, inv_focal, pp, illum_coeffs, kij, kd, Iij, Id);
        float r3 = compute_r3(Dij, Dl, Dr, Du, Dd, i, j, inv_focal, pp, ws);
        float r4 = compute_r4(Dij, Dl, Dr, Du, Dd, i, j, inv_focal, pp, ws);
        float r5 = compute_r5(Dij, Dl, Dr, Du, Dd, i, j, inv_focal, pp, ws);
        float r6 = compute_r6(Dij, initial_Dij);

        Fd(k*num_eq + 0) = sqrt_w_grad * r1;
        Fd(k*num_eq + 1) = sqrt_w_grad * r2;
        Fd(k*num_eq + 2) = sqrt_w_smooth * r3;
        Fd(k*num_eq + 3) = sqrt_w_smooth * r4;
        Fd(k*num_eq + 4) = sqrt_w_smooth * r5;
        Fd(k*num_eq + 5) = sqrt_w_depth * r6;
      }

      // r1
      float weight = sqrt_w_grad;
      int row = k*num_eq + 0;

      int col = k;
      float val = compute_dr1dDij(Dij, Dl, Du, Dr, Dru, i, j, illum_coeffs, kij, kr, inv_focal, pp);
      triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));

      if (left > -1)
      {
        col = left;
        val = compute_dr1dDl(Dij, Dl, Du, i, j, illum_coeffs, kij, inv_focal, pp);
        triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));
      }
      if (up > -1)
      {
        col = up;
        val = compute_dr1dDu(Dij, Dl, Du, i, j, illum_coeffs, kij, inv_focal, pp);
        triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));
      }
      if (right > -1)
      {
        col = right;
        val = compute_dr1dDr(Dij, Dr, Dru, i, j, illum_coeffs, kr, inv_focal, pp);
        triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));
      }
      if (rightup > -1)
      {
        col = rightup;
        val = compute_dr1dDru(Dij, Dr, Dru, i, j, illum_coeffs, kr, inv_focal, pp);
        triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));
      }

      // r2
      weight = sqrt_w_grad;
      row = k*num_eq + 1;

      col = k;
      val = compute_dr2dDij(Dij, Dl, Du, Dd, Ddl, i, j, illum_coeffs, kij, kd, inv_focal, pp);
      triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));

      if (left > -1)
      {
        col = left;
        val = compute_dr2dDl(Dij, Dl, Du, i, j, illum_coeffs, kij, inv_focal, pp);
        triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));
      }
      if (up > -1)
      {
        col = up;
        val = compute_dr2dDu(Dij, Dl, Du, i, j, illum_coeffs, kij, inv_focal, pp);
        triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));
      }
      if (down > -1)
      {
        col = down;
        val = compute_dr2dDd(Dij, Dd, Ddl, i, j, illum_coeffs, kd, inv_focal, pp);
        triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));
      }
      if (leftdown > -1)
      {
        col = leftdown;
        val = compute_dr2dDdl(Dij, Dd, Ddl, i, j, illum_coeffs, kd, inv_focal, pp);
        triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));
      }

      // r3
      weight = sqrt_w_smooth;
      row = k*num_eq + 2;

      col = k;
      val = compute_dr3dDij(i, j, inv_focal, pp);
      triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));

      if (left > -1)
      {
        col = left;
        val = compute_dr3dDl(i, j, inv_focal, pp, ws);
        triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));
      }
      if (right > -1)
      {
        col = right;
        val = compute_dr3dDr(i, j, inv_focal, pp, ws);
        triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));
      }
      if (up > -1)
      {
        col = up;
        val = compute_dr3dDu(i, j, inv_focal, pp, ws);
        triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));
      }
      if (down > -1)
      {
        col = down;
        val = compute_dr3dDd(i, j, inv_focal, pp, ws);
        triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));
      }

      // r4
      weight = sqrt_w_smooth;
      row = k*num_eq + 3;

      col = k;
      val = compute_dr4dDij(i, j, inv_focal, pp);
      triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));

      if (left > -1)
      {
        col = left;
        val = compute_dr4dDl(i, j, inv_focal, pp, ws);
        triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));
      }
      if (right > -1)
      {
        col = right;
        val = compute_dr4dDr(i, j, inv_focal, pp, ws);
        triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));
      }
      if (up > -1)
      {
        col = up;
        val = compute_dr4dDu(i, j, inv_focal, pp, ws);
        triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));
      }
      if (down > -1)
      {
        col = down;
        val = compute_dr4dDd(i, j, inv_focal, pp, ws);
        triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));
      }

      // r5
      weight = sqrt_w_smooth;
      row = k*num_eq + 4;

      col = k;
      val = compute_dr5dDij();
      triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));

      if (left > -1)
      {
        col = left;
        val = compute_dr5dDl(ws);
        triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));
      }
      if (right > -1)
      {
        col = right;
        val = compute_dr5dDr(ws);
        triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));
      }
      if (up > -1)
      {
        col = up;
        val = compute_dr5dDu(ws);
        triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));
      }
      if (down > -1)
      {
        col = down;
        val = compute_dr5dDd(ws);
        triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));
      }

      // r6
      weight = sqrt_w_depth;
      row = k*num_eq + 5;

      col = k;
      val = compute_dr6dDij();
      triplets.push_back(Eigen::Triplet<float>(row, col, weight * val));
      
    }

    float err = Fd.dot(Fd);
    std::cout << "E(d): " << err << std::endl;
    
    std::cout << "Set Jd from triplets" << std::endl;
    Jd.setFromTriplets(triplets.begin(), triplets.end());

    std::cout << "compute JdT * Jd & JdT * Fd" << std::endl;
    Eigen::SparseMatrix<float> JdT = Jd.transpose();
    Eigen::SparseMatrix<float> JdTJd = JdT*Jd;
    Eigen::VectorXf _JdTFd = -JdT*Fd;

    std::cout << "decompose JdT*Jd: " << std::endl;
    Eigen::SimplicialCholesky<Eigen::SparseMatrix<float>> cholesky;
    cholesky.compute(JdTJd);

    //Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> cg;
    //cg.compute(JdTJd);

    std::cout << "solve system" << std::endl;
    delta = cholesky.solve(_JdTFd);

    //delta = cg.solve(_JdFd);

    for (size_t k = 0; k < indices.size(); ++k)
    {
      depthf.at<float>(indices[k]) += delta(k);
    }

    cv::Mat radiance;
    compute_radiance(radiance, depthf, indices, illum_coeffs, inv_focal, pp);

    // update albedo
    if(true)
    {
      for (size_t k = 0; k < indices.size(); ++k)
      {
        albedo_grayf_.at<float>(indices[k]) =
          grayf.at<float>(indices[k]) / radiance.at<float>(indices[k]);
      }

      // clamping data
      cv::max(albedo_grayf_, 0.0f, albedo_grayf_);
      cv::min(albedo_grayf_, 1.0f, albedo_grayf_);
    }    
    
    if(false)
    {
      cv::Mat irradf;
      compute_irradiance(irradf, depthf, indices, albedo_grayf_, illum_coeffs, inv_focal, pp);

      cv::Mat irrad;
      irradf.convertTo(irrad, CV_8U, 255.0f);

      cv::imshow("gray", grayf);
      
      cv::imshow("radiance", radiance);

      cv::imshow("albedo", albedo_grayf_);
      
      cv::imshow("irrad", irrad);

      /*cv::Mat irradf_gradx;
      cv::Sobel(irradf, irradf_gradx, CV_32F, 1, 0);

      cv::Mat irradf_abs_gradx;
      cv::convertScaleAbs(irradf_gradx, irradf_abs_gradx, 255.0);

      cv::Mat irradf_grady;
      cv::Sobel(irradf, irradf_grady, CV_32F, 0, 1);

      cv::Mat irradf_abs_grady;
      cv::convertScaleAbs(irradf_grady, irradf_abs_grady, 255.0);
      
      cv::imshow("irrad grad x", irradf_abs_gradx);
      cv::imshow("irrad grad y", irradf_abs_grady);*/

      cv::waitKey();
      cv::destroyAllWindows();
    }
  }

  if(true)
  { // compute final error
    err_grad = 0;
    err_smooth = 0;
    err_depth = 0;

    for (size_t k = 0; k < indices.size(); ++k)
    {
      int i = indices[k] % width;
      int j = indices[k] / width;

      float Dij = depthf.at<float>(j, i);
      float Dl = depthf.at<float>(j, i - 1);
      float Dr = depthf.at<float>(j, i + 1);
      float Du = depthf.at<float>(j - 1, i);
      float Dd = depthf.at<float>(j + 1, i);
      float Dru = depthf.at<float>(j - 1, i + 1);
      float Ddl = depthf.at<float>(j + 1, i - 1);

      float initial_Dij = initial_depthf.at<float>(j, i);

      float kij = albedo_grayf_.at<float>(j, i);
      float kr = albedo_grayf_.at<float>(j, i + 1);
      float kd = albedo_grayf_.at<float>(j + 1, i);

      float Iij = grayf.at<float>(j, i);
      float Ir = grayf.at<float>(j, i + 1);
      float Id = grayf.at<float>(j + 1, i);

      float r1 = compute_r1(Dij, Dl, Du, Dr, Dru, i, j, inv_focal, pp, illum_coeffs, kij, kr, Iij, Ir);
      float r2 = compute_r2(Dij, Dl, Du, Dd, Ddl, i, j, inv_focal, pp, illum_coeffs, kij, kd, Iij, Id);
      float r3 = compute_r3(Dij, Dl, Dr, Du, Dd, i, j, inv_focal, pp, ws);
      float r4 = compute_r4(Dij, Dl, Dr, Du, Dd, i, j, inv_focal, pp, ws);
      float r5 = compute_r5(Dij, Dl, Dr, Du, Dd, i, j, inv_focal, pp, ws);
      float r6 = compute_r6(Dij, initial_Dij);

      err_grad += (r1*r1 + r2*r2);
      err_smooth += (r3*r3 + r4*r4 + r5*r5);
      err_depth += (r6*r6);
    }

    std::cout << "Refined error";
    std::cout << "- E_grad(d): " << err_grad;
    std::cout << ", E_smooth(d): " << err_smooth;
    std::cout << ", E_depth(d): " << err_depth << std::endl;
  }
}

void DepthOptimizer::filter_valid_pixels(
  const cv::Mat& depthf, const cv::Mat& validmap, 
  std::vector<size_t>& indices)
{  
  for (int ir = 1; ir < (depthf.rows - 1); ++ir)
  {
    for (int ic = 1; ic < (depthf.cols - 1); ++ic)
    {
      // check albedo
      if (validmap.at<unsigned char>(ir,ic) == 0)
      {
        continue;
      }

      // check depth
      if (depthf.at<float>(ir, ic) == 0 ||
        depthf.at<float>(ir, ic - 1) == 0 ||
        depthf.at<float>(ir, ic + 1) == 0 ||
        depthf.at<float>(ir - 1, ic) == 0 ||
        depthf.at<float>(ir + 1, ic) == 0 ||
        depthf.at<float>(ir - 1, ic + 1) == 0 ||
        depthf.at<float>(ir + 1, ic - 1) == 0)
      {
        continue;
      }

      size_t index = ir * depthf.cols + ic;

      indices.push_back(index);
    }
  }
  std::cout << "valid pixels: " << indices.size() << std::endl;
}

void DepthOptimizer::compute_irradiance(
  cv::Mat& irradf,
  const cv::Mat& depthf,
  const std::vector<size_t>& valid_pixels,
  const cv::Mat& albedo_grayf, const std::vector<float>& illum_coeffs,
  const cv::Point2f& inv_focal, const cv::Point2f& pp,
  bool clamp_val /*= false*/)
{
  int width = depthf.cols;
  int height = depthf.rows;

  cv::Mat normals;
  compute_normals(depthf, normals, inv_focal, pp);

  irradf.create(height, width, CV_32F);

  for (size_t idx = 0; idx < valid_pixels.size(); ++idx)
  {
    cv::Vec3f nij = normals.at<cv::Vec3f>(valid_pixels[idx]);
    
    float kij = albedo_grayf.at<float>(valid_pixels[idx]);

    float Bij = compute_B(nij, illum_coeffs, kij);

    irradf.at<float>(valid_pixels[idx]) = Bij;
  } 

  // clamping
  if (true)
  {
    cv::min(irradf, 1.0, irradf);
    cv::max(irradf, 0.0, irradf);
  }
}

void DepthOptimizer::compute_radiance(
  cv::Mat& radf, const cv::Mat& depthf, 
  const std::vector<size_t>& valid_pixels, 
  const std::vector<float>& l, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp)
{
  int width = depthf.cols;
  int height = depthf.rows;

  cv::Mat normals;
  compute_normals(depthf, normals, inv_focal, pp);

  radf.create(height, width, CV_32F);

  for (size_t idx = 0; idx < valid_pixels.size(); ++idx)
  {    
    cv::Vec3f normal = normals.at<cv::Vec3f>(valid_pixels[idx]);

    std::vector<float> sh;
    SphericalHarmonics::eval_3_band(normal, sh);

    float rad = l[0] * sh[0] + l[1] * sh[1] + l[2] * sh[2] +
      l[3] * sh[3] + l[4] * sh[4] + l[5] * sh[5] +
      l[6] * sh[6] + l[7] * sh[7] + l[8] * sh[8];
    
    radf.at<float>(valid_pixels[idx]) = rad;
  }
}

cv::Vec3f DepthOptimizer::compute_normal(
  const cv::Mat& depthf,
  const size_t i, const size_t j,
  const cv::Point2f& inv_focal, const cv::Point2f& pp)
{
  float Dij = depthf.at<float>(j, i);
  float Dl = depthf.at<float>(j, i - 1);
  float Du = depthf.at<float>(j - 1, i);

  cv::Vec3f normal = compute_normal(Dij, Dl, Du, i, j, inv_focal, pp);

  return normal;
}

cv::Vec3f DepthOptimizer::compute_normal(
  const float Dij, const float Dl, const float Du,
  const size_t i, const size_t j,
  const cv::Point2f& inv_focal, const cv::Point2f& pp)
{
  cv::Vec3f normal = compute_normal_unnormalized(Dij, Dl, Du, i, j, inv_focal, pp);

  normal = cv::normalize(normal);

  return normal;
}

cv::Vec3f DepthOptimizer::compute_normal_unnormalized(
  const float Dij, const float Dl, const float Du,
  const size_t i, const size_t j,
  const cv::Point2f& inv_focal, const cv::Point2f& pp)
{
  cv::Vec3f normal;
  normal[0] = Du*(Dij - Dl)*inv_focal.y;
  normal[1] = Dl*(Dij - Du)*inv_focal.x;
  normal[2] = normal[0] * (pp.x - i)*inv_focal.x
    + normal[1] * (pp.y - j)*inv_focal.y
    - Dl*Du*inv_focal.x*inv_focal.y;

  return normal;
}

void DepthOptimizer::compute_normals(
  const cv::Mat& depthf, cv::Mat& normals,
  const cv::Point2f& inv_focal, const cv::Point2f& pp)
{
  int width  = depthf.cols;
  int height = depthf.rows;

  normals.create(height, width, CV_32FC3);

  static cv::Vec3f nan(
    std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::quiet_NaN());

  for (int ir = 1; ir < depthf.rows; ++ir)
  {
    for (int ic = 1; ic < depthf.cols; ++ic)
    {
      if (depthf.at<float>(ir, ic) == 0 ||
        depthf.at<float>(ir, ic - 1) == 0 ||
        depthf.at<float>(ir - 1, ic) == 0)
      {
        normals.at<cv::Vec3f>(ir, ic) = nan;
        continue;
      }

      normals.at<cv::Vec3f>(ir, ic) = 
        compute_normal(depthf, ic, ir, inv_focal, pp);
    }
  }
}

float DepthOptimizer::compute_B(const cv::Vec3f& n, const std::vector<float>& l, const float k)
{
  std::vector<float> sh;
  SphericalHarmonics::eval_3_band(n, sh);

  float B = k*(l[0] * sh[0] + l[1] * sh[1] + l[2] * sh[2] + 
    l[3] * sh[3] + l[4] * sh[4] + l[5] * sh[5] + 
    l[6] * sh[6] + l[7] * sh[7] + l[8] * sh[8]);

  return B;
}

float DepthOptimizer::compute_r1(
  const float Dij, 
  const float Dl, const float Du, const float Dr, const float Dru, 
  const size_t i, const size_t j,
  const cv::Point2f& inv_focal, const cv::Point2f& pp, 
  const std::vector<float>& l, 
  const float kij, const float kr, const float Iij, const float Ir)
{
  cv::Vec3f nij = compute_normal(Dij, Dl, Du, i, j, inv_focal, pp);
  cv::Vec3f nr = compute_normal(Dr, Dij, Dru, i + 1, j, inv_focal, pp);

  float Bij = compute_B(nij, l, kij);
  float Br = compute_B(nr, l, kr);

  float r1 = ((Bij - Br) - (Iij - Ir));

  return r1;
}

float DepthOptimizer::compute_r2(
  const float Dij, 
  const float Dl, const float Du, const float Dd, const float Ddl, 
  const size_t i, const size_t j, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp, 
  const std::vector<float>& l, 
  const float kij, const float kd, 
  const float Iij, const float Id)
{
  cv::Vec3f nij = compute_normal(Dij, Dl, Du, i, j, inv_focal, pp);
  cv::Vec3f nd = compute_normal(Dd, Ddl, Dij, i, j, inv_focal, pp);

  float Bij = compute_B(nij, l, kij);
  float Bd = compute_B(nd, l, kd);

  float r2 = ((Bij - Bd) - (Iij - Id));

  return r2;
}

float DepthOptimizer::compute_r3(
  const float Dij, 
  const float Dl, const float Dr, const float Du, const float Dd,
  const size_t i, const size_t j, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp, float ws /*= 0.25f*/)
{
  const float pij_x = (i - pp.x)*inv_focal.x*Dij;
  const float pl_x  = ((i - 1) - pp.x)*inv_focal.x*Dl;
  const float pr_x  = ((i + 1) - pp.x)*inv_focal.x*Dr;
  const float pu_x  = (i - pp.x)*inv_focal.x*Du;  
  const float pd_x  = (i - pp.x)*inv_focal.x*Dd;

  float r3 = pij_x - ws * (pl_x + pu_x + pr_x + pd_x);

  return r3;
}

float DepthOptimizer::compute_r4(
  const float Dij, const float Dl, const float Dr, const float Du, const float Dd, 
  const size_t i, const size_t j, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp, float ws /*= 0.25f*/)
{
  const float pij_y = (j - pp.y)*inv_focal.y*Dij;
  const float pl_y = (j - pp.y)*inv_focal.y*Dl;
  const float pr_y = (j - pp.y)*inv_focal.y*Dr;
  const float pu_y = ((j - 1) - pp.y)*inv_focal.y*Du;
  const float pd_y = ((j + 1) - pp.y)*inv_focal.y*Dd;

  float r4 = pij_y - ws * (pl_y + pu_y + pr_y + pd_y);

  return r4;
}

float DepthOptimizer::compute_r5(
  const float Dij, const float Dl, const float Dr, const float Du, const float Dd, 
  const size_t i, const size_t j, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp, float ws /*= 0.25f*/)
{  
  float r5 = Dij - ws * (Dl + Du + Dr + Dd);

  return r5;
}

float DepthOptimizer::compute_r6(const float Dij, const float initial_Dij)
{
  float r6 = (Dij - initial_Dij);

  return r6;
}

cv::Vec3f DepthOptimizer::compute_dBdn(
  const cv::Vec3f& n, 
  const std::vector<float>& l, 
  const float k)
{
  cv::Vec3f dBdn;
  dBdn[0] = k*(l[3] + l[4] * n[1] - 2 * l[6] * n[0] + l[7] * n[2] + 2 * l[8] * n[0]);
  dBdn[1] = k*(l[1] + l[4] * n[0] + l[5] * n[2] - 2 * l[6] * n[1] - 2 * l[8] * n[1]);
  dBdn[2] = k*(l[2] + l[5] * n[1] + 4 * l[6] * n[2] + l[7] * n[0]);

  return dBdn;
}

std::vector<cv::Vec3f> DepthOptimizer::compute_dndn_til(
  const cv::Vec3f& n, const cv::Vec3f& n_hat)
{
  //static const double eps = 1e-30;

  float nx2 = n_hat[0] * n_hat[0];
  float ny2 = n_hat[1] * n_hat[1];
  float nz2 = n_hat[2] * n_hat[2];

  float nxny = n_hat[0] * n_hat[1];
  float nynz = n_hat[1] * n_hat[2];
  float nznx = n_hat[2] * n_hat[0];

  float f = nx2 + ny2 + nz2;
  
  float inv_f = 1.0f/f;
  float sqrt_inv_f = std::sqrt(inv_f);
  float sqrt_inv_f3 = inv_f * sqrt_inv_f;
  
  // dnx/dn_hat
  cv::Vec3f dnxdn_hat;
  dnxdn_hat[0] = sqrt_inv_f - nx2 * sqrt_inv_f3;
  dnxdn_hat[1] = -nxny * sqrt_inv_f3;
  dnxdn_hat[2] = -nznx * sqrt_inv_f3;

  // dny/dn_hat
  cv::Vec3f dnydn_hat;
  dnydn_hat[0] = -nxny * sqrt_inv_f3;
  dnydn_hat[1] = sqrt_inv_f - ny2 * sqrt_inv_f3;
  dnydn_hat[2] = -nynz * sqrt_inv_f3;

  // dny/dn_hat
  cv::Vec3f dnzdn_hat;
  dnzdn_hat[0] = -nznx * sqrt_inv_f3;
  dnzdn_hat[1] = -nynz * sqrt_inv_f3;
  dnzdn_hat[2] = sqrt_inv_f - nz2 * sqrt_inv_f3; 

  std::vector<cv::Vec3f> dndn_hat(3);
  dndn_hat[0] = dnxdn_hat;
  dndn_hat[1] = dnydn_hat;
  dndn_hat[2] = dnzdn_hat;

  return dndn_hat;
}

cv::Vec3f DepthOptimizer::compute_dn_tildDij(
  const float Dij, const float Dl, const float Du,
  const size_t i, const size_t j, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp)
{
  cv::Vec3f dn_hatdDij;
  dn_hatdDij[0] = Du * inv_focal.y;
  dn_hatdDij[1] = Dl * inv_focal.x;
  dn_hatdDij[2] = (Du*(pp.x - i) + Dl*(pp.y - j)) * inv_focal.x * inv_focal.y;

  return dn_hatdDij;
}

cv::Vec3f DepthOptimizer::compute_dn_tildDl(
  const float Dij, const float Dl, const float Du,
  const size_t i, const size_t j,
  const cv::Point2f& inv_focal, const cv::Point2f& pp)
{
  cv::Vec3f dn_hatdDl;
  dn_hatdDl[0] = -Du * inv_focal.y;
  dn_hatdDl[1] = (Dij - Du) * inv_focal.x;
  dn_hatdDl[2] = (-Du*(pp.x - i) + (Dij - Du)*(pp.y - j) - Du) * inv_focal.x * inv_focal.y;

  return dn_hatdDl;
}

cv::Vec3f DepthOptimizer::compute_dn_tildDu(
  const float Dij, const float Dl, const float Du,
  const size_t i, const size_t j, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp)
{
  cv::Vec3f dn_hatdDu;
  dn_hatdDu[0] = (Dij - Dl) * inv_focal.y;
  dn_hatdDu[1] = -Dl * inv_focal.x;
  dn_hatdDu[2] = ((Dij - Dl)*(pp.x - i) - Dl*(pp.y - j) - Dl) * inv_focal.x * inv_focal.y;

  return dn_hatdDu;
}

float DepthOptimizer::compute_dr1dDij(/* D(i,j) */ 
  const float Dij, 
  const float Dl, const float Du, 
  const float Dr, const float Dru, 
  const size_t i, const size_t j, 
  const std::vector<float>& l, const float kij, const float kr, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp)
{
  cv::Vec3f nij_tild = compute_normal_unnormalized(Dij, Dl, Du, i, j, inv_focal, pp);
  cv::Vec3f nr_tild = compute_normal_unnormalized(Dr, Dij, Dru, i + 1, j, inv_focal, pp);

  cv::Vec3f nij = cv::normalize(nij_tild);
  cv::Vec3f nr = cv::normalize(nr_tild);

  cv::Vec3f dBdn = compute_dBdn(nij, l, kij);
  cv::Vec3f dBdnr = compute_dBdn(nr, l, kr);

  std::vector<cv::Vec3f> dnijdnij_til = compute_dndn_til(nij, nij_tild);
  std::vector<cv::Vec3f> dnrdnr_til = compute_dndn_til(nr, nr_tild);
  
  cv::Vec3f dn_tildDij = compute_dn_tildDij(Dij, Dl, Du, i, j, inv_focal, pp);
  cv::Vec3f dnr_tildDij = compute_dn_tildDl(Dr, Dij, Dru, i + 1, j, inv_focal, pp); 
  
  cv::Vec3f dndDij;
  dndDij[0] = dnijdnij_til[0].dot(dn_tildDij);
  dndDij[1] = dnijdnij_til[1].dot(dn_tildDij);
  dndDij[2] = dnijdnij_til[2].dot(dn_tildDij);

  cv::Vec3f dnrdDij;
  dnrdDij[0] = dnrdnr_til[0].dot(dnr_tildDij);
  dnrdDij[1] = dnrdnr_til[1].dot(dnr_tildDij);
  dnrdDij[2] = dnrdnr_til[2].dot(dnr_tildDij);

  float dr1dDij = dBdn.dot(dndDij) - dBdnr.dot(dnrdDij);

  return dr1dDij;
}

float DepthOptimizer::compute_dr1dDl(/* D(i-1,j) */
  const float Dij, const float Dl, const float Du,
  const size_t i, const size_t j,
  const std::vector<float>& l, const float kij,
  const cv::Point2f& inv_focal, const cv::Point2f& pp)
{
  cv::Vec3f nij_til = compute_normal_unnormalized(Dij, Dl, Du, i, j, inv_focal, pp);

  cv::Vec3f nij = cv::normalize(nij_til);

  cv::Vec3f dBdn = compute_dBdn(nij, l, kij);

  std::vector<cv::Vec3f> dnijdnij_til = compute_dndn_til(nij, nij_til);
  
  cv::Vec3f dn_tildDl = compute_dn_tildDl(Dij, Dl, Du, i, j, inv_focal, pp);
  
  cv::Vec3f dndDl;
  dndDl[0] = dnijdnij_til[0].dot(dn_tildDl);
  dndDl[1] = dnijdnij_til[1].dot(dn_tildDl);
  dndDl[2] = dnijdnij_til[2].dot(dn_tildDl);

  float dr1dDl = dBdn.dot(dndDl);

  return dr1dDl;
}

float DepthOptimizer::compute_dr1dDu(/* D(i,j-1) */ 
  const float Dij, const float Dl, const float Du, 
  const size_t i, const size_t j, 
  const std::vector<float>& l, const float kij, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp)
{
  cv::Vec3f nij_hat = compute_normal_unnormalized(Dij, Dl, Du, i, j, inv_focal, pp);

  cv::Vec3f nij = cv::normalize(nij_hat);

  cv::Vec3f dBdn = compute_dBdn(nij, l, kij);

  std::vector<cv::Vec3f> dnijdnij_hat = compute_dndn_til(nij, nij_hat);
  
  cv::Vec3f dn_hatdDu = compute_dn_tildDu(Dij, Dl, Du, i, j, inv_focal, pp);

  cv::Vec3f dndDu;
  dndDu[0] = dnijdnij_hat[0].dot(dn_hatdDu);
  dndDu[1] = dnijdnij_hat[1].dot(dn_hatdDu);
  dndDu[2] = dnijdnij_hat[2].dot(dn_hatdDu);

  float dr1dDu = dBdn.dot(dndDu);

  return dr1dDu;
}

float DepthOptimizer::compute_dr1dDr(/* D(i+1,j) */ 
  const float Dij, const float Dr, const float Dru, 
  const size_t i, const size_t j, 
  const std::vector<float>& l, const float kr, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp)
{
  cv::Vec3f nr_til = compute_normal_unnormalized(Dr, Dij, Dru, i + 1, j, inv_focal, pp);

  cv::Vec3f nr = cv::normalize(nr_til);

  cv::Vec3f dBdnr = compute_dBdn(nr, l, kr);

  std::vector<cv::Vec3f> dnrdnr_til = compute_dndn_til(nr, nr_til);

  cv::Vec3f dnr_tildDr = compute_dn_tildDij(Dr, Dij, Dru, i + 1, j, inv_focal, pp);

  cv::Vec3f dnrdDr;
  dnrdDr[0] = dnrdnr_til[0].dot(dnr_tildDr);
  dnrdDr[1] = dnrdnr_til[1].dot(dnr_tildDr);
  dnrdDr[2] = dnrdnr_til[2].dot(dnr_tildDr);

  float dr1dDr = -dBdnr.dot(dnrdDr);

  return dr1dDr;
}

float DepthOptimizer::compute_dr1dDru(/* D(i+1,j-1) */ 
  const float Dij, const float Dr, const float Dru, 
  const size_t i, const size_t j, 
  const std::vector<float>& l, const float kr, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp)
{
  cv::Vec3f nr_til = compute_normal_unnormalized(Dr, Dij, Dru, i + 1, j, inv_focal, pp);

  cv::Vec3f nr = cv::normalize(nr_til);

  cv::Vec3f dBdnr = compute_dBdn(nr, l, kr);

  std::vector<cv::Vec3f> dnrdnr_til = compute_dndn_til(nr, nr_til);

  cv::Vec3f dnr_tildDru = compute_dn_tildDu(Dr, Dij, Dru, i + 1, j, inv_focal, pp);

  cv::Vec3f dnrdDru;
  dnrdDru[0] = dnrdnr_til[0].dot(dnr_tildDru);
  dnrdDru[1] = dnrdnr_til[1].dot(dnr_tildDru);
  dnrdDru[2] = dnrdnr_til[2].dot(dnr_tildDru);

  float dr1dij = -dBdnr.dot(dnrdDru);

  return dr1dij;
}

float DepthOptimizer::compute_dr2dDij(/* D(i,j) */ 
  const float Dij, 
  const float Dl, const float Du, const float Dd, const float Ddl, 
  const size_t i, const size_t j, 
  const std::vector<float>& l, const float kij, const float kd, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp)
{
  cv::Vec3f nij_til = compute_normal_unnormalized(Dij, Dl, Du, i, j, inv_focal, pp);
  cv::Vec3f nd_til = compute_normal_unnormalized(Dd, Ddl, Dij, i, j + 1, inv_focal, pp);

  cv::Vec3f nij = cv::normalize(nij_til);
  cv::Vec3f nd = cv::normalize(nd_til);

  cv::Vec3f dBdn = compute_dBdn(nij, l, kij);
  cv::Vec3f dBdnd = compute_dBdn(nd, l, kd);

  std::vector<cv::Vec3f> dnijdnij_til = compute_dndn_til(nij, nij_til);
  std::vector<cv::Vec3f> dnddnd_til = compute_dndn_til(nd, nd_til);

  cv::Vec3f dn_tildDij = compute_dn_tildDij(Dij, Dl, Du, i, j, inv_focal, pp);
  cv::Vec3f dnd_tildDij = compute_dn_tildDu(Dd, Ddl, Dij, i, j + 1, inv_focal, pp);

  cv::Vec3f dndDij;
  dndDij[0] = dnijdnij_til[0].dot(dn_tildDij);
  dndDij[1] = dnijdnij_til[1].dot(dn_tildDij);
  dndDij[2] = dnijdnij_til[2].dot(dn_tildDij);

  cv::Vec3f dnddDij;
  dnddDij[0] = dnddnd_til[0].dot(dnd_tildDij);
  dnddDij[1] = dnddnd_til[1].dot(dnd_tildDij);
  dnddDij[2] = dnddnd_til[2].dot(dnd_tildDij);

  float dr2dDij = dBdn.dot(dndDij) - dBdnd.dot(dnddDij);

  return dr2dDij;
}

float DepthOptimizer::compute_dr2dDl(/* D(i-1,j) */
  const float Dij, const float Dl, const float Du,
  const size_t i, const size_t j,
  const std::vector<float>& l, const float kij,
  const cv::Point2f& inv_focal, const cv::Point2f& pp)
{
  float dr2dDl = compute_dr1dDl(Dij, Dl, Du, i, j, l, kij, inv_focal, pp);

  return dr2dDl;
}

float DepthOptimizer::compute_dr2dDu(/* D(i,j-1) */ 
  const float Dij, const float Dl, const float Du, 
  const size_t i, const size_t j, 
  const std::vector<float>& l, const float kij, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp)
{
  float dr2dDu = compute_dr1dDu(Dij, Dl, Du, i, j, l, kij, inv_focal, pp);

  return dr2dDu;
}

float DepthOptimizer::compute_dr2dDd(/* D(i,j+1) */ 
  const float Dij, const float Dd, const float Ddl, 
  const size_t i, const size_t j, 
  const std::vector<float>& l, const float kd, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp)
{
  cv::Vec3f nd_til = compute_normal_unnormalized(Dd, Ddl, Dij, i, j + 1, inv_focal, pp);

  cv::Vec3f nd = cv::normalize(nd_til);

  cv::Vec3f dBdnd = compute_dBdn(nd, l, kd);

  std::vector<cv::Vec3f> dnddnd_til = compute_dndn_til(nd, nd_til);

  cv::Vec3f dnd_tildDd = compute_dn_tildDij(Dd, Ddl, Dij, i, j + 1, inv_focal, pp);

  cv::Vec3f dnddDd;
  dnddDd[0] = dnddnd_til[0].dot(dnd_tildDd);
  dnddDd[1] = dnddnd_til[1].dot(dnd_tildDd);
  dnddDd[2] = dnddnd_til[2].dot(dnd_tildDd);
  
  float dr2dDij = -dBdnd.dot(dnddDd);

  return dr2dDij;
}

float DepthOptimizer::compute_dr2dDdl(/* D(i-1,j+1) */ 
  const float Dij, const float Dd, const float Ddl, 
  const size_t i, const size_t j, 
  const std::vector<float>& l, const float kd,
  const cv::Point2f& inv_focal, const cv::Point2f& pp)
{
  cv::Vec3f nd_til = compute_normal_unnormalized(Dd, Ddl, Dij, i, j + 1, inv_focal, pp);

  cv::Vec3f nd = cv::normalize(nd_til);

  cv::Vec3f dBdnd = compute_dBdn(nd, l, kd);

  std::vector<cv::Vec3f> dnddnd_til = compute_dndn_til(nd, nd_til);

  cv::Vec3f dnd_tildDdl = compute_dn_tildDl(Dd, Ddl, Dij, i, j + 1, inv_focal, pp);

  cv::Vec3f dnddDdl;
  dnddDdl[0] = dnddnd_til[0].dot(dnd_tildDdl);
  dnddDdl[1] = dnddnd_til[1].dot(dnd_tildDdl);
  dnddDdl[2] = dnddnd_til[2].dot(dnd_tildDdl);

  float dr2dDij = -dBdnd.dot(dnddDdl);

  return dr2dDij;
}

float DepthOptimizer::compute_dr3dDij(/* D(i,j) */ 
  const size_t i, const size_t j, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp)
{
  return ((i - pp.x) * inv_focal.x);
}

float DepthOptimizer::compute_dr3dDl(/* D(i-1,j) */ 
  const size_t i, const size_t j, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp, 
  const float ws /*= 0.25f*/)
{
  return (-ws * ((i - 1) - pp.x) * inv_focal.x);
}

float DepthOptimizer::compute_dr3dDu(/* D(i,j-1) */ 
  const size_t i, const size_t j, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp, 
  const float ws /*= 0.25f*/)
{
  return (-ws * (i - pp.x) * inv_focal.x);
}

float DepthOptimizer::compute_dr3dDr(/* D(i+1,j) */ 
  const size_t i, const size_t j, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp, 
  const float ws /*= 0.25f*/)
{
  return (-ws * ((i + 1) - pp.x) * inv_focal.x);
}

float DepthOptimizer::compute_dr3dDd(/* D(i,j+1) */ 
  const size_t i, const size_t j, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp, 
  const float ws /*= 0.25f*/)
{
  return (-ws * (i - pp.x) * inv_focal.x);
}

float DepthOptimizer::compute_dr4dDij(/* D(i,j) */ 
  const size_t i, const size_t j, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp)
{
  return ((j - pp.y) * inv_focal.y);
}

float DepthOptimizer::compute_dr4dDl(/* D(i-1,j) */ 
  const size_t i, const size_t j, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp, 
  const float ws /*= 0.25f*/)
{
  return (-ws * (j - pp.y) * inv_focal.y);
}

float DepthOptimizer::compute_dr4dDu(/* D(i,j-1) */ 
  const size_t i, const size_t j, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp, 
  const float ws /*= 0.25f*/)
{
  return (-ws * ((j - 1) - pp.y) * inv_focal.y);
}

float DepthOptimizer::compute_dr4dDr(/* D(i+1,j) */ 
  const size_t i, const size_t j, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp, 
  const float ws /*= 0.25f*/)
{
  return (-ws * (j - pp.y) * inv_focal.y);
}

float DepthOptimizer::compute_dr4dDd(/* D(i,j+1) */ 
  const size_t i, const size_t j, 
  const cv::Point2f& inv_focal, const cv::Point2f& pp, 
  const float ws /*= 0.25f*/)
{
  return (-ws * ((j + 1) - pp.y) * inv_focal.y);
}

float DepthOptimizer::compute_dr5dDij()
{
  return 1.0f;
}

float DepthOptimizer::compute_dr5dDl(const float ws /*= 0.25f*/)
{
  return (-ws);
}

float DepthOptimizer::compute_dr5dDu(const float ws /*= 0.25f*/)
{
  return (-ws);
}

float DepthOptimizer::compute_dr5dDr(const float ws /*= 0.25f*/)
{
  return (-ws);
}

float DepthOptimizer::compute_dr5dDd(const float ws /*= 0.25f*/)
{
  return (-ws);
}

float DepthOptimizer::compute_dr6dDij()
{
  return 1.0f;
}

void DepthOptimizer::test_derivative()
{
  float Dij = 0.597198f;
  float Dl = 0.596605f; 
  float Du = 0.596198f;
  float Dr = 0.597605f; 
  float Dru = 0.597f;
  float Dd = 0.597605f; 
  float Ddl = 0.596605f;
  
  size_t i = 470;
  size_t j = 202;  
  
  cv::Point2f focal(574.053f, 574.053f);
  cv::Point2f pp(320.0f, 240.0f);
  cv::Point2f inv_focal(1 / focal.x, 1 / focal.y);

  float kij = 0.655149f;
  float kr = 0.581969f;
  float kd = 0.603257f;
  
  float Iij = 0.352941f;
  float Ir = 0.341176f; 
  float Id = 0.372549f;

  std::vector<float> l(9);
  l[0] = 0.760564f;
  l[1] = -0.038657f;
  l[2] = 0.0916755f;
  l[3] = -0.112711f;
  l[4] = 0.0111526f;
  l[5] = 0.143617f;
  l[6] = 0.0588849f;
  l[7] = -0.00743658f;
  l[8] = 0.0274434f;
  
  float delta = 0.00001f;
  
  // r1
  float r1 = compute_r1(Dij, Dl, Du, Dr, Dru, i, j, inv_focal, pp, l, kij, kr, Iij, Ir);

  // dr1/dDij
  float analytic_dr1dDij = compute_dr1dDij(Dij, Dl, Du, Dr, Dru, i, j, l, kij, kr, inv_focal, pp);
  
  float r1_Dij_delta = compute_r1(Dij + delta, Dl, Du, Dr, Dru, i, j, inv_focal, pp, l, kij, kr, Iij, Ir);  
  float numerical_dr1dDij = (r1_Dij_delta - r1) / delta;

  // dr1/dDl
  float analytic_dr1dDl = compute_dr1dDl(Dij, Dl, Du, i, j, l, kij, inv_focal, pp);

  float r1_Dl_delta = compute_r1(Dij, Dl + delta, Du, Dr, Dru, i, j, inv_focal, pp, l, kij, kr, Iij, Ir);  
  float numerical_dr1dDl = (r1_Dl_delta - r1) / delta;

  // dr1/dDl
  float analytic_dr1dDu = compute_dr1dDu(Dij, Dl, Du, i, j, l, kij, inv_focal, pp);

  float r1_Du_delta = compute_r1(Dij, Dl, Du + delta, Dr, Dru, i, j, inv_focal, pp, l, kij, kr, Iij, Ir);
  float numerical_dr1dDu = (r1_Du_delta - r1) / delta;

  // dr1/dDr
  float analytic_dr1dDr = compute_dr1dDr(Dij, Dr, Dru, i, j, l, kr, inv_focal, pp);

  float r1_Dr_delta = compute_r1(Dij, Dl, Du, Dr + delta, Dru, i, j, inv_focal, pp, l, kij, kr, Iij, Ir);
  float numerical_dr1dDr = (r1_Dr_delta - r1) / delta;

  // dr1/dDr
  float analytic_dr1dDru = compute_dr1dDru(Dij, Dr, Dru, i, j, l, kr, inv_focal, pp);

  float r1_Dru_delta = compute_r1(Dij, Dl, Du, Dr, Dru + delta, i, j, inv_focal, pp, l, kij, kr, Iij, Ir);
  float numerical_dr1dDru = (r1_Dru_delta - r1) / delta;

  std::cout << "--------------------------------------------" <<std::endl;
  std::cout << "analytic  dr1/dDij  " << analytic_dr1dDij << std::endl;
  std::cout << "numerical dr1/dDij  " << numerical_dr1dDij << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr1/dDl   " << analytic_dr1dDl << std::endl;
  std::cout << "numerical dr1/dDl   " << numerical_dr1dDl << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr1/dDu   " << analytic_dr1dDu << std::endl;
  std::cout << "numerical dr1/dDu   " << numerical_dr1dDu << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr1/dDr   " << analytic_dr1dDr << std::endl;
  std::cout << "numerical dr1/dDr   " << numerical_dr1dDr << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr1/dDru  " << analytic_dr1dDru << std::endl;
  std::cout << "numerical dr1/dDru  " << numerical_dr1dDru << std::endl;
  std::cout << "--------------------------------------------" << std::endl;

  // r2
  float r2 = compute_r2(Dij, Dl, Du, Dd, Ddl, i, j, inv_focal, pp, l, kij, kd, Iij, Id);

  // dr2/dDij
  float analytic_dr2dDij = compute_dr2dDij(Dij, Dl, Du, Dd, Ddl, i, j, l, kij, kd, inv_focal, pp);

  float r2_Dij_delta = compute_r2(Dij + delta, Dl, Du, Dd, Ddl, i, j, inv_focal, pp, l, kij, kd, Iij, Id);  
  float numerical_dr2dDij = (r2_Dij_delta - r2) / delta;

  // dr2/dDl
  float analytic_dr2dDl = compute_dr2dDl(Dij, Dl, Du, i, j, l, kij, inv_focal, pp);

  float r2_Dl_delta = compute_r2(Dij, Dl + delta, Du, Dd, Ddl, i, j, inv_focal, pp, l, kij, kd, Iij, Id);
  float numerical_dr2dDl = (r2_Dl_delta - r2) / delta;

  // dr2/dDu
  float analytic_dr2dDu = compute_dr2dDu(Dij, Dl, Du, i, j, l, kij, inv_focal, pp);

  float r2_Du_delta = compute_r2(Dij, Dl, Du + delta, Dd, Ddl, i, j, inv_focal, pp, l, kij, kd, Iij, Id);
  float numerical_dr2dDu = (r2_Du_delta - r2) / delta;

  // dr2/dDd
  float analytic_dr2dDd = compute_dr2dDd(Dij, Dd, Ddl, i, j, l, kd, inv_focal, pp);

  float r2_Dd_delta = compute_r2(Dij, Dl, Du, Dd + delta, Ddl, i, j, inv_focal, pp, l, kij, kd, Iij, Id);
  float numerical_dr2dDd = (r2_Dd_delta - r2) / delta;

  // dr2/dDdl
  float analytic_dr2dDdl = compute_dr2dDdl(Dij, Dd, Ddl, i, j, l, kd, inv_focal, pp);

  float r2_Ddl_delta = compute_r2(Dij, Dl, Du, Dd, Ddl + delta, i, j, inv_focal, pp, l, kij, kd, Iij, Id);
  float numerical_dr2dDdl = (r2_Ddl_delta - r2) / delta;

  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr2/dDij  " << analytic_dr2dDij << std::endl;
  std::cout << "numerical dr2/dDij  " << numerical_dr2dDij << std::endl;  
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr2/dDl   " << analytic_dr2dDl << std::endl;
  std::cout << "numerical dr2/dDl   " << numerical_dr2dDl << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr2/dDu   " << analytic_dr2dDu << std::endl;
  std::cout << "numerical dr2/dDu   " << numerical_dr2dDu << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr2/dDd   " << analytic_dr2dDd << std::endl;
  std::cout << "numerical dr2/dDd   " << numerical_dr2dDd << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr2/dDdl  " << analytic_dr2dDdl << std::endl;
  std::cout << "numerical dr2/dDdl  " << numerical_dr2dDdl << std::endl;
  std::cout << "--------------------------------------------" << std::endl;

  // r3
  float r3 = compute_r3(Dij, Dl, Dr, Du, Dd, i, j, inv_focal, pp);

  // dr3/dDij
  float analytic_dr3dDij = compute_dr3dDij(i, j, inv_focal, pp);

  float r3_Dij_delta = compute_r3(Dij + delta, Dl, Dr, Du, Dd, i, j, inv_focal, pp);
  float numerical_dr3dDij = (r3_Dij_delta - r3) / delta;

  // dr3/dDl
  float analytic_dr3dDl = compute_dr3dDl(i, j, inv_focal, pp);

  float r3_Dl_delta = compute_r3(Dij, Dl + delta, Dr, Du, Dd, i, j, inv_focal, pp);
  float numerical_dr3dDl = (r3_Dl_delta - r3) / delta;

  // dr3/dDr
  float analytic_dr3dDr = compute_dr3dDr(i, j, inv_focal, pp);

  float r3_Dr_delta = compute_r3(Dij, Dl, Dr + delta, Du, Dd, i, j, inv_focal, pp);
  float numerical_dr3dDr = (r3_Dr_delta - r3) / delta;

  // dr3/dDu
  float analytic_dr3dDu = compute_dr3dDu(i, j, inv_focal, pp);

  float r3_Du_delta = compute_r3(Dij, Dl, Dr, Du + delta, Dd, i, j, inv_focal, pp);
  float numerical_dr3dDu = (r3_Du_delta - r3) / delta;

  // dr3/dDd
  float analytic_dr3dDd = compute_dr3dDd(i, j, inv_focal, pp);

  float r3_Dd_delta = compute_r3(Dij, Dl, Dr, Du, Dd + delta, i, j, inv_focal, pp);
  float numerical_dr3dDd = (r3_Dd_delta - r3) / delta;

  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr3/dDij  " << analytic_dr3dDij << std::endl;
  std::cout << "numerical dr3/dDij  " << numerical_dr3dDij << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr3/dDl   " << analytic_dr3dDl << std::endl;
  std::cout << "numerical dr3/dDl   " << numerical_dr3dDl << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr3/dDr   " << analytic_dr3dDr << std::endl;
  std::cout << "numerical dr3/dDr   " << numerical_dr3dDr << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr3/dDu   " << analytic_dr3dDu << std::endl;
  std::cout << "numerical dr3/dDu   " << numerical_dr3dDu << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr3/dDd   " << analytic_dr3dDd << std::endl;
  std::cout << "numerical dr3/dDd   " << numerical_dr3dDd << std::endl;
  std::cout << "--------------------------------------------" << std::endl;

  // r4
  float r4 = compute_r4(Dij, Dl, Dr, Du, Dd, i, j, inv_focal, pp);

  // dr4/dDij
  float analytic_dr4dDij = compute_dr4dDij(i, j, inv_focal, pp);

  float r4_Dij_delta = compute_r4(Dij + delta, Dl, Dr, Du, Dd, i, j, inv_focal, pp);
  float numerical_dr4dDij = (r4_Dij_delta - r4) / delta;

  // dr4/dDl
  float analytic_dr4dDl = compute_dr4dDl(i, j, inv_focal, pp);

  float r4_Dl_delta = compute_r4(Dij, Dl + delta, Dr, Du, Dd, i, j, inv_focal, pp);
  float numerical_dr4dDl = (r4_Dl_delta - r4) / delta;

  // dr4/dDr
  float analytic_dr4dDr = compute_dr4dDr(i, j, inv_focal, pp);

  float r4_Dr_delta = compute_r4(Dij, Dl, Dr + delta, Du, Dd, i, j, inv_focal, pp);
  float numerical_dr4dDr = (r4_Dr_delta - r4) / delta;

  // dr4/dDu
  float analytic_dr4dDu = compute_dr4dDu(i, j, inv_focal, pp);

  float r4_Du_delta = compute_r4(Dij, Dl, Dr, Du + delta, Dd, i, j, inv_focal, pp);
  float numerical_dr4dDu = (r4_Du_delta - r4) / delta;

  // dr4/dDd
  float analytic_dr4dDd = compute_dr4dDd(i, j, inv_focal, pp);

  float r4_Dd_delta = compute_r4(Dij, Dl, Dr, Du, Dd + delta, i, j, inv_focal, pp);
  float numerical_dr4dDd = (r4_Dd_delta - r4) / delta;

  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr4/dDij  " << analytic_dr4dDij << std::endl;
  std::cout << "numerical dr4/dDij  " << numerical_dr4dDij << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr4/dDl   " << analytic_dr4dDl << std::endl;
  std::cout << "numerical dr4/dDl   " << numerical_dr4dDl << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr4/dDr   " << analytic_dr4dDr << std::endl;
  std::cout << "numerical dr4/dDr   " << numerical_dr4dDr << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr4/dDu   " << analytic_dr4dDu << std::endl;
  std::cout << "numerical dr4/dDu   " << numerical_dr4dDu << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr4/dDd   " << analytic_dr4dDd << std::endl;
  std::cout << "numerical dr4/dDd   " << numerical_dr4dDd << std::endl;
  std::cout << "--------------------------------------------" << std::endl;

  // r5
  float r5 = compute_r5(Dij, Dl, Dr, Du, Dd, i, j, inv_focal, pp);

  // dr5/dDij
  float analytic_dr5dDij = compute_dr5dDij();

  float r5_Dij_delta = compute_r5(Dij + delta, Dl, Dr, Du, Dd, i, j, inv_focal, pp);
  float numerical_dr5dDij = (r5_Dij_delta - r5) / delta;

  // dr5/dDl
  float analytic_dr5dDl = compute_dr5dDl();

  float r5_Dl_delta = compute_r5(Dij, Dl + delta, Dr, Du, Dd, i, j, inv_focal, pp);
  float numerical_dr5dDl = (r5_Dl_delta - r5) / delta;

  // dr5/dDr
  float analytic_dr5dDr = compute_dr5dDr();

  float r5_Dr_delta = compute_r5(Dij, Dl, Dr + delta, Du, Dd, i, j, inv_focal, pp);
  float numerical_dr5dDr = (r5_Dr_delta - r5) / delta;

  // dr5/dDu
  float analytic_dr5dDu = compute_dr5dDu();

  float r5_Du_delta = compute_r5(Dij, Dl, Dr, Du + delta, Dd, i, j, inv_focal, pp);
  float numerical_dr5dDu = (r5_Du_delta - r5) / delta;

  // dr5/dDd
  float analytic_dr5dDd = compute_dr5dDd();

  float r5_Dd_delta = compute_r5(Dij, Dl, Dr, Du, Dd + delta, i, j, inv_focal, pp);
  float numerical_dr5dDd = (r5_Dd_delta - r5) / delta;

  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr5/dDij  " << analytic_dr5dDij << std::endl;
  std::cout << "numerical dr5/dDij  " << numerical_dr5dDij << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr5/dDl   " << analytic_dr5dDl << std::endl;
  std::cout << "numerical dr5/dDl   " << numerical_dr5dDl << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr5/dDr   " << analytic_dr5dDr << std::endl;
  std::cout << "numerical dr5/dDr   " << numerical_dr5dDr << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr5/dDu   " << analytic_dr5dDu << std::endl;
  std::cout << "numerical dr5/dDu   " << numerical_dr5dDu << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr5/dDd   " << analytic_dr5dDd << std::endl;
  std::cout << "numerical dr5/dDd   " << numerical_dr5dDd << std::endl;
  std::cout << "--------------------------------------------" << std::endl;

  // r6
  float r6 = compute_r6(Dij, Dij);

  // dr6/dDij
  float analytic_dr6dDij = compute_dr6dDij();

  float r6_Dij_delta = compute_r6(Dij + delta, Dij);

  float numerical_dr6dDij = (r6_Dij_delta - r6) / delta;

  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "analytic  dr6/dDij  " << analytic_dr6dDij << std::endl;
  std::cout << "numerical dr6/dDij  " << numerical_dr6dDij << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  
}

