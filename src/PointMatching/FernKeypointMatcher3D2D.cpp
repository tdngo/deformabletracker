//=========================================================================
// Author       :  Ngo Tien Dat
// Email        :  dat.ngo@epfl.ch
// Organization :  EPFL
// Purpose      :  Fern keypoint matcher to match the input image and the
//                 reference image
// Date         :  10 May 2012
//=========================================================================

#include "FernKeypointMatcher3D2D.h"
#include <Utils/DUtils.h>

using namespace arma;

FernKeypointMatcher3D2D::FernKeypointMatcher3D2D(
  const IplImage* referenceImgRGB,
  const cv::Point& topLeft,          const cv::Point& topRight,
  const cv::Point& bottomRight,      const cv::Point& bottomLeft,
  const TriangleMesh& referenceMesh, const Camera& camCamera,
  const string& modelImageFile )
  : KeypointMatcher3D2D(referenceImgRGB, topLeft, topRight, bottomRight, bottomLeft, referenceMesh, camCamera)
{
//  range.set_range_variation_for_theta(-datum::pi/6, datum::pi/6);     // [-PI/6, PI/6]  - ROTATION
  range.set_range_variation_for_theta(0, 2*datum::pi);                  // [0, 2*PI]    - ROTATION
  range.set_range_variation_for_phi(datum::pi/4, 3*datum::pi/4);        // [PI/4, 3PI/4]  - SLANT
  range.independent_scaling(0.8, 1.2, 0.8, 1.2);                        // [0.8, 1.2]    - SCALE

  // Save roi of reference image into a file and use fern detector builder to load
  assert(cvSaveImage((modelImageFile + "_temporary.png").c_str(), referenceImgRGB));
  ofstream roif((modelImageFile + ".roi").c_str());
  assert(roif.good());
  roif << topLeft.x   << " " << topLeft.y     << endl
     << topRight.x    << " " << topRight.y    << endl
     << bottomRight.x << " " << bottomRight.y << endl
     << bottomLeft.x  << " " << bottomLeft.y  << endl;
  roif.close();

  detector = planar_pattern_detector_builder::build_with_cache(modelImageFile.c_str(),
                       &range,
                       500,        // Number of keypoints
                       10000,
                       0.0,
                       32, 7, 4,
                       30, 12,
                       100000, 2000);

  detector->set_maximum_number_of_points_to_detect(2000);

  // Pre-compute 3D coordinates of model keypoints
  existed3DRefKeypoints.resize(detector->number_of_model_points);
  bary3DRefKeypoints.set_size(detector->number_of_model_points, 6);

  for(int i = 0; i < detector->number_of_model_points; i++)
  {
    cv::Point2d refPoint( detector->model_points[i].fr_u(), detector->model_points[i].fr_v() );
    rowvec intersectPoint;
    bool isIntersect = this->find3DPointOnMesh(refPoint, intersectPoint);

    existed3DRefKeypoints[i] = isIntersect;
    if (isIntersect)
    {
      this->bary3DRefKeypoints(i, span(0,5)) = intersectPoint;
    }
  }
}

FernKeypointMatcher3D2D::~FernKeypointMatcher3D2D()
{
  delete detector;
}

int FernKeypointMatcher3D2D::matchImages2D2D(IplImage* inputImageGray)
{
  mat matchesSort(detector->number_of_model_points, 5);

  detector->detect(inputImageGray);
  int number_of_matches = 0;
  for(int i = 0; i < detector->number_of_model_points; i++)
  {
    if (detector->model_points[i].class_score > 0)
    {
      match1Data[2*number_of_matches]   = detector->model_points[i].fr_u();
      match1Data[2*number_of_matches + 1]  = detector->model_points[i].fr_v();

      match2Data[2*number_of_matches]    = detector->model_points[i].potential_correspondent->fr_u();
      match2Data[2*number_of_matches + 1]  = detector->model_points[i].potential_correspondent->fr_v();

      matchesSort(number_of_matches, 0)  = detector->model_points[i].class_score;
      matchesSort(number_of_matches, 1)  = detector->model_points[i].fr_u();
      matchesSort(number_of_matches, 2)  = detector->model_points[i].fr_v();
      matchesSort(number_of_matches, 3)  = detector->model_points[i].potential_correspondent->fr_u();
      matchesSort(number_of_matches, 4)  = detector->model_points[i].potential_correspondent->fr_v();

      number_of_matches++;
    }
  }

  // Sort matches in decreasing order of score
  matchesSort.resize(number_of_matches, 5);
  uvec indices = sort_index( matchesSort.col(0), 1 );

  for (int i = 0; i < number_of_matches; ++i)
  {
    match1Data[2*number_of_matches]   = matchesSort(indices(i), 1);
    match1Data[2*number_of_matches + 1]  = matchesSort(indices(i), 2);
    match2Data[2*number_of_matches]    = matchesSort(indices(i), 3);
    match2Data[2*number_of_matches + 1]  = matchesSort(indices(i), 4);
  }

  return number_of_matches;
}

arma::mat FernKeypointMatcher3D2D::MatchImages3D2D(IplImage* inputImageRGB)
{
  // Convert color to gray
  assert(inputImageRGB->nChannels == 3);
  IplImage* inputImageGray = cvCreateImage(cvGetSize(inputImageRGB), inputImageRGB->depth, 1);
  cvConvertImage(inputImageRGB, inputImageGray, CV_BGR2GRAY);

#if 0
  // ================ For visualization of matches only ================
  int n2D2DMatches = this->matchImages2D2D(inputImageGray);    // Result is stored in match1Data, match2Data
  arma::mat matches2D2D(n2D2DMatches, 4);
  for (int i = 0; i < n2D2DMatches; i++)
  {
    matches2D2D(i, 0) = match1Data[2*i];
    matches2D2D(i, 1) = match1Data[2*i+1];

    matches2D2D(i, 2) = match2Data[2*i];
    matches2D2D(i, 3) = match2Data[2*i+1];
  }
  Visualization::VisualizeMatches(this->referenceImgRGB, inputImageRGB, matches2D2D, 10);
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#endif

  // Do matching
  detector->detect(inputImageGray);
  cvReleaseImage(&inputImageGray);

  int nMatches = 0;
  // For every model point
  for(int i = 0; i < detector->number_of_model_points; i++)
  {
    if (detector->model_points[i].class_score > 0 && existed3DRefKeypoints[i])
    {
      matches3D2D(nMatches, span(0,5)) = this->bary3DRefKeypoints.row(i);

      matches3D2D(nMatches, 6) = detector->model_points[i].potential_correspondent->fr_u();
      matches3D2D(nMatches, 7) = detector->model_points[i].potential_correspondent->fr_v();

      // ID of the 3D point of the match
      matches3D2D(nMatches, 8) = i;
      nMatches++;
    }
  }

  if (nMatches > 0)
    return matches3D2D.rows(0, nMatches-1);
  else
    return arma::mat(0, matches3D2D.n_cols);
}


