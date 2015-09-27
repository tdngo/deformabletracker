//////////////////////////////////////////////////////////////////////////
// Author       :  Ngo Tien Dat
// Email        :  dat.ngo@epfl.ch
// Organization :  EPFL
// Purpose      :  For visualization purposes like drawing a mesh on image
// Date         :  19 March 2012
//////////////////////////////////////////////////////////////////////////

#include "Visualization.h"
#include "DefinedMacros.h"

using namespace arma;
using namespace std;
using namespace cv;

void Visualization::DrawProjectedMesh(cv::Mat inputImg, 
                                      const TriangleMesh& triangleMesh, 
                                      const Camera& camera,
                                      const cv::Scalar& color)
{
  mat projPoints = camera.ProjectPoints(triangleMesh.GetVertexCoords());

  // Drawing mesh on image
  const umat& edges = triangleMesh.GetEdges();
  for (int i = 0; i < triangleMesh.GetNEdges(); i++)
  {
    int vid1 = edges(0, i);    // Vertex id 1
    int vid2 = edges(1, i);    // Vertex id 2

    cv::Point2d pt1(projPoints(vid1,0), projPoints(vid1,1));
    cv::Point2d pt2(projPoints(vid2,0), projPoints(vid2,1));

    // Draw a line
    cv::line(inputImg, pt1, pt2, color);
  }
}

void Visualization::DrawPointsAsDot(cv::Mat inputImg, const mat& points, const cv::Scalar& color)
{
  for (unsigned int i = 0; i < points.n_rows; i++)
  {
    cv::Point2d aPoint(points(i,0), points(i,1));     // Convert to opencv Point object
    int radius = 2;

    cv::circle(inputImg, aPoint, radius, color, -1);  // -1 mean filled circle
  }
}

void Visualization::DrawPointsAsPlus(cv::Mat image, const mat& points, const cv::Scalar& color)
{
  int len = 3;
  for (unsigned int i = 0 ; i < points.n_rows; i++)
  {
    line(image, cvPoint(points(i,0) - len, points(i,1)), cvPoint(points(i,0) + len, points(i,1)), color);
    line(image, cvPoint(points(i,0), points(i,1) - len), cvPoint(points(i,0), points(i,1) + len), color);
  }        
}

void Visualization::DrawVectorOfPointsAsPlus(cv::Mat image, const vector<cv::Point2f>& points, const cv::Scalar& color)
{
  int len = 3;
  for (unsigned int i = 0 ; i < points.size(); i++)
  {
    line(image, cvPoint(points[i].x - len, points[i].y), cvPoint(points[i].x + len, points[i].y), color);
    line(image, cvPoint(points[i].x, points[i].y - len), cvPoint(points[i].x, points[i].y + len), color);
  }
}

void Visualization::DrawVectorOfKeypointsAsPlus(cv::Mat image, const vector<cv::KeyPoint>& points, const cv::Scalar& color)
{
  int len = 3;  // Half length of plus sign
  for (unsigned int i = 0 ; i < points.size(); i++)
  {
    line(image, cvPoint(points[i].pt.x - len, points[i].pt.y), cvPoint(points[i].pt.x + len, points[i].pt.y), color);
    line(image, cvPoint(points[i].pt.x, points[i].pt.y - len), cvPoint(points[i].pt.x, points[i].pt.y + len), color);
  }
}

void Visualization::DrawAQuadrangle(cv::Mat image, 
                                    const cv::Point& topLeft, const cv::Point& topRight, 
                                    const cv::Point& bottomRight, const cv::Point& bottomLeft,
                                    const CvScalar& color)
{
  line(image, topLeft, topRight, color);
  line(image, topRight, bottomRight, color);
  line(image, bottomRight, bottomLeft, color);
  line(image, bottomLeft, topLeft, color);
}

void Visualization::DrawMatches(const cv::Mat& image1, const cv::Mat& image2, const arma::mat& matches, cv::Mat& outImage)
{
  cv::Size size(image1.cols + image2.cols, MAX(image1.rows, image2.rows));
  outImage.create(size, CV_MAKETYPE(image1.depth(), 3));    // Existing memory may be reused
  outImage = cv::Scalar::all(0);

  cv::Mat outImg1 = outImage( cv::Rect(0, 0, image1.cols, image1.rows) );
  cv::Mat outImg2 = outImage( cv::Rect(image1.cols, 0, image2.cols, image2.rows) );

  if (image1.type() == CV_8U)
    cvtColor(image1, outImg1, CV_GRAY2BGR);
  else
    image1.copyTo(outImg1);  // Deep copy

  if (image2.type() == CV_8U)
    cvtColor(image2, outImg2, CV_GRAY2BGR);
  else
    image2.copyTo(outImg2);  // Deep copy

  Visualization::DrawPointsAsPlus(outImg1, matches.cols(0,1), KPT_COLOR);
  Visualization::DrawPointsAsPlus(outImg2, matches.cols(2,3), KPT_COLOR);

  // Draw matches on the output image
  int nMatches = matches.n_rows;
  for (int i = 0; i < nMatches; i++)
  {
    line(outImage,
         cv::Point(matches(i,0), matches(i,1)),
         cv::Point(matches(i,2) + image1.cols, matches(i,3)),
         MATCH_COLOR);
  }
}

void Visualization::VisualizeMatches(const cv::Mat& img1, const vector<KeyPoint>& keypoints1,
                                     const cv::Mat& img2, const vector<KeyPoint>& keypoints2,
                                     const vector<DMatch>& matches1to2,
                                     int   nMatchesDrawn)
{
  int nMatches = matches1to2.size();

  vector<DMatch> randMatches1to2 = matches1to2;
  //std::random_shuffle(randMatches1to2.begin(), randMatches1to2.end());

  cv::Mat matchImage;
  for (int i = 0; i < nMatches; i += nMatchesDrawn)
  {
    vector<KeyPoint> someKpts1;
    vector<KeyPoint> someKpts2;
    vector<DMatch>   someDMatches;
    for (int j = 0; j < nMatchesDrawn && i+j < nMatches; ++j)
    {
      DMatch aMatch = randMatches1to2[i+j];

      someKpts1.push_back(keypoints1[aMatch.queryIdx]);
      someKpts2.push_back(keypoints2[aMatch.trainIdx]);

      aMatch.queryIdx = j;
      aMatch.trainIdx = j;
      someDMatches.push_back(aMatch);
    }
    cv::drawMatches(img1, someKpts1, img2, someKpts2, someDMatches, matchImage);

    imshow("Visualize some matches at a time - Press anykey to continue!", matchImage);
    if (nMatchesDrawn != nMatches && waitKey() == 27)
      return;
  }
}

void Visualization::VisualizeMatches(const cv::Mat& img1, const cv::Mat& img2,
                                     const arma::mat& matches, int nMatchesDrawn)
{
  vector<KeyPoint> keypoints1;
  vector<KeyPoint> keypoints2;
  vector<DMatch>   matches1to2;

  KeyPoint aKeypoint1;
  KeyPoint aKeypoint2;
  DMatch   aDMatch;
  for (unsigned int i = 0; i < matches.n_rows; i++)
  {
    aKeypoint1.pt = Point2f(matches(i,0), matches(i,1));
    aKeypoint2.pt = Point2f(matches(i,2), matches(i,3));
    aDMatch       = DMatch(i,i,0);

    keypoints1.push_back(aKeypoint1);
    keypoints2.push_back(aKeypoint2);
    matches1to2.push_back(aDMatch);
  }
  Visualization::VisualizeMatches(img1, keypoints1, img2, keypoints2, matches1to2, nMatchesDrawn);
}

void Visualization::VisualizeBaryMatches(const cv::Mat& refImage, 
                                         const cv::Mat& inputImg, 
                                         const Camera& camCamera, 
                                         const TriangleMesh& refeMesh, 
                                         const mat& baryMatches)
{
  if (baryMatches.n_rows == 0)
    return;

  mat points3D     = ConvertBary3DTo3D(baryMatches.cols(0,5), refeMesh);
  mat projPoints   = camCamera.ProjectPoints( points3D );
  mat imagePoints  = baryMatches.cols(6,7);

  mat matches2D2D  = join_rows(projPoints, imagePoints);

  Visualization::VisualizeMatches(refImage, inputImg, matches2D2D, matches2D2D.n_rows);
}

mat Visualization::ConvertBary3DTo3D(const mat& pointsBary3D, const TriangleMesh& referenceMesh)
{
  const mat& vertexCoords = referenceMesh.GetVertexCoords();

  int nMatches = pointsBary3D.n_rows;
  mat points3D(nMatches, 3);

  int    vId1, vId2, vId3;
  double b1, b2, b3;

  for(int i = 0; i < nMatches; i++)
  {
    vId1 = (int)pointsBary3D(i, 0);
    vId2 = (int)pointsBary3D(i, 1);
    vId3 = (int)pointsBary3D(i, 2);

    // 3D vertex coordinates
    const rowvec& vertex1Coords = vertexCoords.row(vId1);
    const rowvec& vertex2Coords = vertexCoords.row(vId2);
    const rowvec& vertex3Coords = vertexCoords.row(vId3);

    b1 = pointsBary3D(i, 3);
    b2 = pointsBary3D(i, 4);
    b3 = pointsBary3D(i, 5);

    // 3D point
    points3D.row(i) = b1*vertex1Coords + b2*vertex2Coords + b3*vertex3Coords;
  }

  return points3D;
}



