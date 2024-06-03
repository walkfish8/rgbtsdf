/**
 * Copyright (c) 2023, Li Yunqiang, walkfish8@hotmail.com.
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the organization nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ''AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTOR BE
 * LIABLE  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include <stdio.h>
#include <stack>
#include <chrono>
#include <limits>
#include <future>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <functional>
#include <opencv2/opencv.hpp>

#define RGBTSDF_USE_EIGEN
#define RGBTSDF_PRINT_VERBOSE
#define RGBTSDF_USE_BGR_COLOR
// #define RGBTSDF_RESTRAIN_THIN_OBJECT
#include "rgbtsdf.hpp"

namespace rgbtsdf {
static inline void Loadrts(const char* path, std::vector<cv::Mat>& RTs) {
  FILE* f = NULL;
  RGBTSDF_FOPEN(f, path, "r");
  if (!f) return;
  int num = 0;
  RGBTSDF_FSCANF(f, "%d\n", &num);
  if (!num) return;
  RTs.resize(num);
  double a1, a2, a3, b1, b2, b3, c1, c2, c3, tx, ty, tz, d1, d2, d3, d4;
  for (int i = 0; i < num; ++i) {
    RGBTSDF_FSCANF(
        f, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
        &a1, &a2, &a3, &tx, &b1, &b2, &b3, &ty, &c1, &c2, &c3, &tz, &d1, &d2,
        &d3, &d4);
    // cv::Mat RT = cv::Mat(3, 4, CV_64F);
    cv::Mat RT = (cv::Mat_<double>(4, 4) << a1, a2, a3, tx, b1, b2, b3, ty, c1,
                  c2, c3, tz, d1, d2, d3, d4);
    // cv::Rodrigues(rvec, R);
    /*R.copyTo(RT.colRange(0, 3));
    cv::Mat T = (cv::Mat_<double>(3, 1) << tx, ty, tz);
    T.copyTo(RT.rowRange(0, 3).colRange(3, 4));*/

    RTs[i] = RT;
  }
  fclose(f);
}
static inline void LoadRTs(const char* path, std::vector<cv::Mat>& RTs) {
  FILE* f = NULL;
  RGBTSDF_FOPEN(f, path, "r");
  if (!f) return;
  int num = 0;
  RGBTSDF_FSCANF(f, "%d\n", &num);
  if (!num) return;
  RTs.resize(num);
  double rx, ry, rz, tx, ty, tz;
  for (int i = 0; i < num; ++i) {
    RGBTSDF_FSCANF(f, "%lf %lf %lf %lf %lf %lf", &rx, &ry, &rz, &tx, &ty, &tz);
    cv::Mat R, RT = cv::Mat(3, 4, CV_64F);
    cv::Mat rvec = (cv::Mat_<double>(3, 1) << rx, ry, rz);
    cv::Rodrigues(rvec, R);
    R.copyTo(RT.colRange(0, 3));
    cv::Mat T = (cv::Mat_<double>(3, 1) << tx, ty, tz);
    T.copyTo(RT.rowRange(0, 3).colRange(3, 4));

    RTs[i] = RT;
  }
  fclose(f);
}

static inline void depthToPoints(const cv::Mat& img, const cv::Mat& K,
                                 std::vector<cv::Point3f>& points) {
  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);

  int ptnum = cv::countNonZero(img);
  if (!ptnum) return;
  points.resize(ptnum);
  ptnum = 0;
  for (int i = 0; i < img.rows; ++i) {
    const ushort* imgptr = img.ptr<unsigned short>(i);
    for (int j = 0; j < img.cols; ++j) {
      if (!imgptr[j]) continue;
      double z = imgptr[j] / 64.0f;
      points[ptnum].x = (float)((j - cx) * z / fx);
      points[ptnum].y = (float)((i - cy) * z / fy);
      points[ptnum++].z = (float)z;
    }
  }
}
}  // namespace rgbtsdf

int main(int argc, char** argv) {
  std::string input_dir = R"(C:\data\test)";
  std::string output_dir = input_dir;

  // load data
  std::vector<cv::Mat> RTs;
  rgbtsdf::Loadrts(R"(C:\data\test\final_pose.txt)", RTs);
  std::vector<cv::String> names;
  cv::glob(R"(C:\data\test\*.png)", names, false);
  std::vector<cv::String> rgbnames;
  cv::glob(R"(C:\data\test\*.jpg)", rgbnames, false);

  // main paramters
  float depthScale = 1 / 64.0f;
  float nearZ = 0;
  float farZ = 1000.0;
  double voxelUnit = 4.0;
  double truncatedUnit = voxelUnit * 8;
  double fx = 2534.069580 / 2;
  double fy = 2533.682861 / 2;
  double cx = 689.968872 / 2;
  double cy = 518.446960 / 2;

  rgbtsdf::Point3_<double> centerPt = {0, 0, 0};
  cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

  // Test CXX API Function.
  using RGBTSFOctree =
      rgbtsdf::OcTree_<10, 8, rgbtsdf::TSDFType::TINYTSDF, float>;
  using RGBTSDFPointMap = RGBTSFOctree::_MapPointData;
  RGBTSFOctree octree(voxelUnit, centerPt);
  octree.setTruncatedDistance(truncatedUnit);

  RGBTSDFPointMap mapPoints;
  // names.resize(30);
  auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < (int)names.size(); ++i) {
    printf("%d\n", i);
    cv::Mat img = cv::imread(names[i], CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat depth;
    img.convertTo(depth, CV_32F, depthScale);
    img = cv::imread(rgbnames[i], CV_LOAD_IMAGE_COLOR);
    cv::resize(img, img, depth.size());

    cv::Mat RT = cv::Mat::eye(4, 4, CV_64F);
    RTs[i].copyTo(RT.rowRange(0, 4));
    RT = RT.inv();

    octree.integrateDepthImage(K, RT, nearZ, farZ, depth, img);

    // cv::Mat rasterDepth = cv::Mat::zeros(depth.size(), CV_32F);
    // cv::Mat rasterImage = cv::Mat::zeros(depth.size(), CV_8UC3);
    // octree.rasterDepthImage({fx, fy, cx, cy}, RT.ptr<double>(), 100,
    //     200, {rasterDepth.data, rasterDepth.cols, rasterDepth.rows},
    //     {rasterImage.data, rasterImage.cols, rasterImage.rows});

    // std::vector<rgbtsdf::Point3_<float>> points;
    // rgbtsdf::depthToPoints<float>(
    //    {rasterDepth.data, depth.cols, depth.rows},
    //    (double(*)[3])K.data, points);
    // rgbtsdf::RT_<float>(RTs[i].ptr<double>())
    //    .transform(points.data(), points.data(), points.size());

    // if (false && i % 10 == 0) {
    //     std::vector<rgbtsdf::Point3_<float>> points, normals;
    //     for (auto iter = mapPoints.begin(); iter != mapPoints.end();
    //          ++iter) {
    //         auto& point_vec = std::get<0>(iter->second);
    //         auto& color_vec = std::get<1>(iter->second);
    //         points.insert(
    //             points.end(), point_vec.begin(), point_vec.end());
    //         // normals.insert(
    //         //     normals.end(), normal_vec.begin(),
    //         normal_vec.end());
    //     }
    //     // char buf[256];
    //     // sprintf_s(buf, R"(C:\data\1\%04d.asc)", i);
    //     // rgbtsdf::writePointsToASC(buf, (float(*)[3])points.data(),
    //     //    (float(*)[3])normals.data(), points.size());
    // }
  }
  auto end = std::chrono::steady_clock::now();
  printf("%f\n",
         std::chrono::duration_cast<std::chrono::duration<float>>(end - begin)
             .count());

  std::vector<rgbtsdf::RGBPixel> triPixels;
  std::vector<rgbtsdf::Point3_<int>> triIndexs;
  std::vector<rgbtsdf::Point3_<float>> triPoints;
  std::vector<rgbtsdf::Point3_<float>> triNormals;

  octree.extractPointCloud(triPoints, triPixels, 0.1);
  octree.extractNormals(triPoints, triNormals);

  char buf[256];
  sprintf_s(buf, "%s\\tsdf.asc", output_dir.c_str());
  rgbtsdf::writePointsToASC(
      buf, triPoints.data(),
      triPoints.size() == triNormals.size() ? triNormals.data() : nullptr,
      triPoints.size() == triPixels.size() ? triPixels.data() : nullptr,
      triPoints.size());

  triPoints.clear(), triPixels.clear();

  std::unordered_map<size_t, int> pointMap[3];
  octree.extractTriMeshs(triPoints, triPixels, triIndexs, 1, pointMap);
  octree.refineTSDFOrientedPoint(triPoints.data(), triPixels.data(),
                                 triPoints.size(), triIndexs.data(),
                                 triIndexs.size(), pointMap, 10);
  octree.extractNormals(triPoints, triNormals);

  sprintf_s(buf, "%s\\tsdf.obj", output_dir.c_str());
  rgbtsdf::writeTriMeshToOBJ(
      buf, triPoints.data(),
      triPoints.size() == triNormals.size() ? triNormals.data() : nullptr,
      triPoints.size() == triPixels.size() ? triPixels.data() : nullptr,
      triPoints.size(), triIndexs.data(), triIndexs.size());
  return 0;
}