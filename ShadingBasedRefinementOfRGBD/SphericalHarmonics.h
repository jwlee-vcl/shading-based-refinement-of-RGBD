#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

class SphericalHarmonics
{
public:
  static void eval_3_band(
    const cv::Vec3f& normal, std::vector<float>& sh)
  {
    sh.resize(9);

    /* l = 0 */
    sh[0] = 1.0f;

    /* l = 1 */
    sh[1] = normal[1];
    sh[2] = normal[2];
    sh[3] = normal[0];

    /* l = 2 */
    sh[4] = normal[0]*normal[1];
    sh[5] = normal[1]*normal[2];
    sh[6] = -normal[0]*normal[0] - normal[1]*normal[1] + 2.0f*normal[2]*normal[2];

    sh[7] = normal[0]*normal[2];
    sh[8] = normal[0]*normal[0] - normal[1]*normal[1];
  }

  static void derivative_3_band(
    const cv::Vec3f& normal, std::vector<cv::Vec3f>& sh_deriv)
  {
    sh_deriv.resize(9);

    /* sh 0 */
    sh_deriv[0] = 0.0f;
    sh_deriv[1] = 0.0f;
    sh_deriv[2] = 0.0f;

    /* sh 1 */
    sh_deriv[3] = 0.0f;
    sh_deriv[4] = 1.0f;
    sh_deriv[5] = 0.0f;

    /* sh 2 */
    sh_deriv[6] = 0.0f;
    sh_deriv[7] = 0.0f;
    sh_deriv[8] = 1.0f;

    /* sh 3 */
    sh_deriv[9]  = 1.0f;
    sh_deriv[10] = 0.0f;
    sh_deriv[11] = 0.0f;

    /* sh 4 */
    sh_deriv[12] = normal[1];
    sh_deriv[13] = normal[0];
    sh_deriv[14] = 0.0f;

    /* sh 5 */
    sh_deriv[15] = 0.0f;
    sh_deriv[16] = normal[2];
    sh_deriv[17] = normal[1];

    /* sh 6 */
    sh_deriv[18] = -2.0f * normal[0];
    sh_deriv[19] = -2.0f * normal[1];
    sh_deriv[20] = 4.0f * normal[2];

    /* sh 7 */
    sh_deriv[21] = normal[2];
    sh_deriv[22] = 0.0f;
    sh_deriv[23] = normal[0];

    /* sh 8 */
    sh_deriv[24] = 2.0f * normal[0];
    sh_deriv[25] = -2.0f * normal[1];
    sh_deriv[26] = 0.0f;
  }

  static void eval_4_band(
    const cv::Vec3f& normal, std::vector<float>& sh)
  {
    eval_3_band(normal, sh);

    sh.resize(16);

    float nx2 = normal[0] * normal[0];
    float ny2 = normal[1] * normal[1];
    float nz2 = normal[2] * normal[2];

    /* l = 3 */
    sh[9] = (3.0f * nx2 - ny2) * normal[1];
    sh[10] = normal[0] * normal[1] * normal[2];
    sh[11] = (4.0f * nz2 - nx2 - ny2) * normal[1];
    sh[12] = (2.0f * nz2 - 3.0f * nx2 - 3.0f * ny2) * normal[2];
    sh[13] = (4.0f * nz2 - nx2 - ny2) * normal[0];
    sh[14] = (nx2 - ny2) * normal[2];
    sh[15] = (nx2 - 3.0f * ny2) * normal[0];
  }

  static void derivative_4_band(
    const cv::Vec3f& normal, std::vector<cv::Vec3f>& sh_deriv)
  {
    derivative_3_band(normal, sh_deriv);

    sh_deriv.resize(16);

    float nx2 = normal[0] * normal[0];
    float ny2 = normal[1] * normal[1];
    float nz2 = normal[2] * normal[2];

    /* sh 9 */
    sh_deriv[27] = 6.0f * normal[0] * normal[1];
    sh_deriv[28] = 3.0f * (nx2 - ny2);
    sh_deriv[29] = 0.0f;

    /* sh 10 */
    sh_deriv[30] = normal[1] * normal[2];
    sh_deriv[31] = normal[0] * normal[2];
    sh_deriv[32] = normal[0] * normal[1];

    /* sh 11 */
    sh_deriv[33] = -2.0f * normal[0] * normal[1];
    sh_deriv[34] =  4.0f * nz2 - nx2 - 3.0f * ny2;
    sh_deriv[35] =  8.0f * normal[1] * normal[2];

    /* sh 12 */
    sh_deriv[36] = -6.0f * normal[0] * normal[2];
    sh_deriv[37] = -6.0f * normal[1] * normal[2];
    sh_deriv[38] =  6.0f * nz2 - 3.0f * (nx2 + ny2);

    /* sh 13 */
    sh_deriv[39] =  4.0f * nz2 - 3.0f * nx2 - ny2;
    sh_deriv[40] = -2.0f * normal[0] * normal[1];
    sh_deriv[41] =  8.0f * normal[0] * normal[2];

    /* sh 14 */
    sh_deriv[42] =  2.0f * normal[0] * normal[2];
    sh_deriv[43] = -2.0f * normal[1] * normal[2];
    sh_deriv[44] = nx2 - ny2;

    /* sh 15 */
    sh_deriv[45] =  3.0f * (nx2 - ny2);
    sh_deriv[46] = -6.0f * normal[0] * normal[1];
    sh_deriv[47] =  0.0f;

  }
};
