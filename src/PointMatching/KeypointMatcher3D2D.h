//=========================================================================
// Author		:	Ngo Tien Dat
// Email		:	dat.ngo@epfl.ch
// Organization	:	EPFL
// 
// Purpose		:	Keypoint matcher that matches the input image and the 
//					reference image. Points in the input image are 2D, points 
//					on reference image is 3D
//					
// Date			:	23 April 2012
//=========================================================================

#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <armadillo>
#include <Mesh/TriangleMesh.h>
#include <Utils/DUtils.h>
#include <Camera.h>

class KeypointMatcher3D2D
{

public:
	// Maximum number of keypoint matches allowed by the program
	static const int MAXIMUM_NUMBER_OF_MATCHES = 2000;

protected:
	// Reference image in RGB color space and inferred gray scale
	IplImage* referenceImgRGB;
	IplImage* referenceImgGray;

	// The part of the referenceImgGray, which is the template object drawn by the user.
	IplImage* templateImageGray;

	// Four corners of the actual template. From these, we can compute ROI mask
	cv::Point topLeft;
	cv::Point topRight;
	cv::Point bottomRight;
	cv::Point bottomLeft;

	// The coordinates of the keypoints matching each other.
	// match1Data represents the keypoint coordinates in the reference image
	// match2Data represents the matching keypoints detected in the input image. 
	// Elements with even indices contain x coordinates and those with odd 
	// indices contain y coordinates of the keypoints detected:
	double match1Data[2 * MAXIMUM_NUMBER_OF_MATCHES];
	double match2Data[2 * MAXIMUM_NUMBER_OF_MATCHES];

	// Matches between 3D points to 2D points. The matrix has 8 or 9 columns
	// [vid1 vid2 vid3 bari1 bari2 bari3 uImage vImage modelPointID]
	// The first three columns represent vertex ids,
	// The next three columns represent barycentric coordinates,
	// The next two columns represent 2D image point in input image.
	// The last column represent the ID of 3D points in reference mesh.
	// There might be more than one match corresponding to the same 3D point
	arma::mat matches3D2D;

private:
	// Reference triangular mesh in CAMERA coordinate that is 3D coordinate 
	// of the reference planar object
	const TriangleMesh &referenceMesh;

	// Camera object in camera coordinate
	const Camera& camCamera;

public:
	// Constructor to be called by inherited classes
	KeypointMatcher3D2D (
					const IplImage* referenceImgRGB, 
					const cv::Point& topLeft,	  const cv::Point& topRight, 
					const cv::Point& bottomRight, const cv::Point& bottomLeft,
					const TriangleMesh& referenceMesh, const Camera& camCamera );

	virtual ~KeypointMatcher3D2D();

	// Find 3D-2D matches between the input image and the reference image.
	// This function will call "MatchImages2D2D" function and convert 2D-2D matches to 3D-2D ones
	// This function can be overriden by directly implementing it, i.e. not call "MatchImages2D2D" function
	// Input:
	//  + input image in RGB color space
	// Output:
	//  + Return the matches in matrix format. The matrix contains 8 or 9 columns
	//    [vid1 vid2 vid3 bari1 bari2 bari3 uImage vImage modelPointID]
	//    The first three columns represent vertex ids
	//    The next three columns represent barycentric coordinates
	//    The last two columns represent 2D image point in input image
	virtual arma::mat MatchImages3D2D( IplImage* inputImageRGB );

private:

	// PURE virtual function to be overridden by the child classes
	// Find 2D-2D matches between the input image and the reference image
	// The input image must be in gray scale. 
	// Match data will be stored in matchData1 and matchData2
	// Return:
	//  + Number of matches
	virtual int matchImages2D2D(IplImage* inputImageGray) = 0;

	// Find intersection between the ray from source to destination and the triangle ABC
	// The problem is formulated as AX = b where X = [lambda1 lambda2 lambda3 alpha]'
	// Input:
	//  + source point
	//  + destination point
	//  + triangle ABC represented by 3D coordinates: 3 rows of [x y z]
	// Output:
	//  + Return the barycentric coordinates of the intersection point
	inline static arma::vec findIntersectionRayTriangle( const arma::vec& source, const arma::vec& destination, const arma::mat& vABC );

protected:
	// Project the image point onto the reference mesh. Shoot a ray from camera center
	// to the 2D point in the reference image. Find intersection between the ray and the mesh
	// Notice the case when the ray may cut the mesh more than once. Select the closest face
	// Input:
	//  + 2D point on the reference image
	//  + reference mesh: attribute of this class
	//  + camera object: attribute of this class
	// Return:
	//  + "true" if there is intersection, "false" otherwise
	//  + Update reference argument "intersectPoint": a row vector with 6 columns [vid1 vid2 vid3 bary1 bary2 bary3]
	bool find3DPointOnMesh( const cv::Point2d& refPoint, arma::rowvec& intersectPoint ) const;

	// Takes keypoints and culls them by the response
	static void retainBest(std::vector<cv::KeyPoint>& keypoints, int n_points) {
		//this is only necessary if the keypoints size is greater than the number of desired points.
		if (n_points > 0 && keypoints.size() > (size_t) n_points) {
			if (n_points == 0) {
				keypoints.clear();
				return;
			}
			//first use nth element to partition the keypoints into the best and worst.
			std::nth_element( keypoints.begin(), keypoints.begin() + n_points, keypoints.end(), KeypointResponseGreater());
			//this is the boundary response, and in the case of FAST may be ambigous
			float ambiguous_response = keypoints[n_points - 1].response;
			//use std::partition to grab all of the keypoints with the boundary response.
			std::vector<cv::KeyPoint>::const_iterator new_end = std::partition( keypoints.begin() + n_points, keypoints.end(), KeypointResponseGreaterThanThreshold(ambiguous_response) );
			//resize the keypoints, given this new end point. nth_element and partition reordered the points inplace
			keypoints.resize(new_end - keypoints.begin());
		}
	}

private:
	struct KeypointResponseGreaterThanThreshold {
		KeypointResponseGreaterThanThreshold(float _value) :
				value(_value) {
		}
		inline bool operator()(const cv::KeyPoint& kpt) const {
			return kpt.response >= value;
		}
		float value;
	};

	struct KeypointResponseGreater {
		inline bool operator()(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2) const {
			return kp1.response > kp2.response;
		}
	};
};









