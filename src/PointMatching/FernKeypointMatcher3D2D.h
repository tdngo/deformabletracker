//=========================================================================
// Author       :  Ngo Tien Dat
// Email        :  dat.ngo@epfl.ch
// Organization :  EPFL
// Purpose      :  FERN keypoint matcher to match the input image and the
//                 reference image
// Date         :  30 April 2012
//=========================================================================

#pragma once

#include "KeypointMatcher3D2D.h"

#include <ferns/mcv.h>
#include <ferns/planar_pattern_detector_builder.h>

class FernKeypointMatcher3D2D : public KeypointMatcher3D2D
{
private:
  affine_transformation_range range;
  planar_pattern_detector * detector;

  // Precomputed 3D bary-centric coordinates of all keypoints in reference image
  arma::mat        bary3DRefKeypoints;
  vector<bool>     existed3DRefKeypoints;    // To indicate a feature points lies on the reference mesh.

public:
  // Constructor
  FernKeypointMatcher3D2D(const IplImage* referenceImgRGB,
              const cv::Point& topLeft,    const cv::Point& topRight, 
              const cv::Point& bottomRight, const cv::Point& bottomLeft,
              const TriangleMesh& referenceMesh, const Camera& camCamera,
              const string& modelImageFile);

  // Destructor
  virtual ~FernKeypointMatcher3D2D();

  // Override the virtual function of the parent class. Compute 3D-2D matches directly.
  // Don't use 3D-2D match function of the parent class to speed up
  virtual arma::mat MatchImages3D2D(IplImage* inputImageRGB);

private:
  // Override the PURE virtual function defined in the base class
  virtual int matchImages2D2D(IplImage* inputImageGray);

};

