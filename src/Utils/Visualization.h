//////////////////////////////////////////////////////////////////////////
// Author		:	Ngo Tien Dat
// Email		:	dat.ngo@epfl.ch
// Organization	:	EPFL
// Purpose		:	For visualization purposes like drawing a mesh on image
// Date			:	19 March 2012
//////////////////////////////////////////////////////////////////////////

#pragma once

#include <armadillo>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <Mesh/LaplacianMesh.h>
#include <Camera.h>

// For visualization purposes like drawing a mesh on image
class Visualization
{
public:

	// Draw the projected mesh vertices on the image
	// Input:
	//  + inputImg: image to draw. Will be changed
	//  + triangleMesh: a triangle mesh that contains edge information
	//  + camera: Camera object
	//  + color to draw the mesh
	// Output:
	//  + inputImg is changed since opencv implementation allow shared data when copying
	static void DrawProjectedMesh( cv::Mat inputImg, const TriangleMesh& triangleMesh,
								   const Camera& camera,
								   const cv::Scalar& color );

	// Draw points on image as dots. Input image will be changed
	// Input:
	//  + Input image
	//  + points to draw: each row of "points" is [x y]
	//  + color to draw
	// Output:
	//  + "image" is changed since opencv implementation allow shared data when copying
	static void DrawPointsAsDot( cv::Mat image, const arma::mat& points, const cv::Scalar& color );

	// Draw a list of points as a plus sign on image
	// Input:
	//  + Input image
	//  + points to draw: each row of "points" is [x y]
	//  + color to draw
	// Output:
	//  + "image" is changed since opencv implementation allow shared data when copying
	static void DrawPointsAsPlus( cv::Mat image, const arma::mat& points, const cv::Scalar& color);

	// Draw a vector of points on image
	static void DrawVectorOfPointsAsPlus(cv::Mat image, const std::vector<cv::Point2f>& points, const cv::Scalar& color);

	// Draw a vector of keypoints on image
	static void DrawVectorOfKeypointsAsPlus(cv::Mat image, const std::vector<cv::KeyPoint>& points, const cv::Scalar& color);

	// Draw a quadrangle specified by four points
	static void DrawAQuadrangle( cv::Mat image,
								 const cv::Point& topLeft,	   const cv::Point& topRight, 
								 const cv::Point& bottomRight, const cv::Point& bottomLeft,
								 const CvScalar& color);

	// Draw matches between two images
	// Input:
	//  + two images: can be either gray or color
	//  + matches: each row contains [x1 y1 x2 y2]
	//	+ outImage: image to draw, memory can be reused
	// Output:
	//  + Update reference argument outImage
	static void DrawMatches(const cv::Mat& image1, const cv::Mat& image2, const arma::mat& matches, cv::Mat& outImage);

	// Draw matches between two images: each time draw just a few matches to see how they are
	// Press anykey to draw next set
	// Input:
	//  + two images with keypoints: can be either gray or color
	//  + matches returned by opencv matching to match img1 to img2
	//	+ nMatchesDrawn: Number of matches to be drawn each time
	// Return:
	//	+ Function terminates if all matches have been visualized
	static void VisualizeMatches( const cv::Mat& img1, const std::vector<cv::KeyPoint>& keypoints1,
					  	  	  	  	  const cv::Mat& img2, const std::vector<cv::KeyPoint>& keypoints2,
					  	  	  	  	  const std::vector<cv::DMatch>& matches1to2,
					  	  	  	  	  int nMatchesDrawn);

	// Draw matches between two images: each time draw just a few matches to see how they are
	// Press anykey to draw next set. Press ESC to stop visualization
	// Input:
	//  + two images with keypoints: can be either gray or color
	//  + matches with each row is [x1 y1 x2 y2]
	//	+ nMatchesDrawn: Number of matches to be drawn each time
	// Return:
	//	+ Function terminates if all matches have been visualized
	static void VisualizeMatches( const cv::Mat& img1, const cv::Mat& img2,
								  const arma::mat& matches, int nMatchesDrawn);

	// Visualize matches in barycentric coordinate
	// Input:
	//		baryMatches		: [vid1 vid2 vid3 b1 b2 b3 u v (modelPointId)]
	static void VisualizeBaryMatches( const cv::Mat& refImage, const cv::Mat& inputImg, const Camera& camCamera, const TriangleMesh& refeMesh, const arma::mat& baryMatches );

	// Convert 3D points in bary-centric coordinate to regular 3D points
	static arma::mat ConvertBary3DTo3D(const arma::mat& pointsBary3D, const TriangleMesh& referenceMesh);

};








