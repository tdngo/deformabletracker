//=========================================================================
// Author		:	Ngo Tien Dat
// Email		:	dat.ngo@epfl.ch
// Organization	:	EPFL
// Purpose		:	Keypoint matcher to match the input image and the 
//					reference image
// Date			:	23 April 2012
//=========================================================================

#include "KeypointMatcher3D2D.h"
#include <Utils/Visualization.h>
#include <timer/Timer.h>
#include <Utils/DefinedMacros.h>

using namespace std;
using namespace arma;

KeypointMatcher3D2D::KeypointMatcher3D2D(
					const IplImage* referenceImgRGB, 
					const cv::Point& topLeft,     const cv::Point& topRight, 
					const cv::Point& bottomRight, const cv::Point& bottomLeft,
					const TriangleMesh& referenceMesh, const Camera& camCamera )

					: topLeft		( topLeft		)
					, topRight		( topRight		)
					, bottomRight	( bottomRight	)
					, bottomLeft	( bottomLeft	)
					, referenceMesh ( referenceMesh )
					, camCamera		( camCamera		)
{
	// Clone the reference image
	CvSize referenceImgSize	 = cvSize(referenceImgRGB->width, referenceImgRGB->height);
	this->referenceImgRGB	 = cvCreateImage(referenceImgSize, referenceImgRGB->depth, referenceImgRGB->nChannels);
	cvCopyImage(referenceImgRGB, this->referenceImgRGB);

	// Generate the gray scale version of reference image
	this->referenceImgGray	 = cvCreateImage(referenceImgSize, IPL_DEPTH_8U, 1);
	cvCvtColor(this->referenceImgRGB, this->referenceImgGray, CV_BGR2GRAY );

	// Calculate the size of the template
	const int templateWidth  = bottomRight.x - topLeft.x;
	const int templateHeight = bottomRight.y - topLeft.y;

	const CvSize templateSize = cvSize(templateWidth, templateHeight);
	const CvRect templateRect = cvRect(topLeft.x, topLeft.y, templateWidth, templateHeight);

	this->templateImageGray	  = cvCreateImage(templateSize, IPL_DEPTH_8U, 1);

	// Store the gray scale version of the template (only ROI)
	cvSetImageROI(this->referenceImgGray, templateRect);
	cvCopyImage(this->referenceImgGray, this->templateImageGray);
	cvResetImageROI(this->referenceImgGray);

	// Initialize matrix "matches3D2D"
	matches3D2D.resize( MAXIMUM_NUMBER_OF_MATCHES, 9 );
}

KeypointMatcher3D2D::~KeypointMatcher3D2D()
{
	cvReleaseImage(&referenceImgRGB);
	cvReleaseImage(&referenceImgGray);
	cvReleaseImage(&templateImageGray);
}

arma::mat KeypointMatcher3D2D::MatchImages3D2D( IplImage* inputImageRGB )
{
	assert(inputImageRGB->nChannels == 3);

	// Match 2D-2D
	IplImage* inputImageGray = cvCreateImage(cvGetSize(inputImageRGB), inputImageRGB->depth, 1);
	cvConvertImage(inputImageRGB, inputImageGray, CV_BGR2GRAY);

	Timer timer;
	timer.start();
	int nMatches = this->matchImages2D2D(inputImageGray);		// Result is stored in match1Data, match2Data
	timer.stop();
	cout << "Number of 2D-2D matches: " << nMatches << endl;
	cout << "Time to match 2D-2D: " << timer.getElapsedTimeInMilliSec() << " ms" << endl;
	cvReleaseImage(&inputImageGray);

	// ----------------------- VISUALIZATION ------------------------------
	arma::mat matches2D2D(nMatches, 4);
	for (int i = 0; i < nMatches; i++)
	{
		matches2D2D(i, 0) = match1Data[2*i];
		matches2D2D(i, 1) = match1Data[2*i+1];

		matches2D2D(i, 2) = match2Data[2*i];
		matches2D2D(i, 3) = match2Data[2*i+1];
	}

	mat imCorners; imCorners.load("data/im_corners.txt");
	int pad = 0;
	cv::Point topLeft		(imCorners(0,0) - pad, imCorners(0,1) - pad);
	cv::Point topRight		(imCorners(1,0) + pad, imCorners(1,1) - pad);
	cv::Point bottomRight	(imCorners(2,0) + pad, imCorners(2,1) + pad);
	cv::Point bottomLeft	(imCorners(3,0) - pad, imCorners(3,1) + pad);
	Visualization::DrawAQuadrangle(referenceImgRGB, topLeft, topRight, bottomRight, bottomLeft, TPL_COLOR);

	Visualization::VisualizeMatches(this->referenceImgRGB, inputImageRGB, matches2D2D, 10);
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	// Transform 2D points on the reference image into 3D points
	timer.start();
	int 		count = 0;
	bool 		isIntersect;
	cv::Point2d refPoint;
	rowvec 		intersectPoint;
	for (int i = 0; i < nMatches; i++)
	{
		refPoint.x = match1Data[2*i];
		refPoint.y = match1Data[2*i+1];
		isIntersect = this->find3DPointOnMesh(refPoint, intersectPoint);

		if (isIntersect)
		{
			this->matches3D2D(count, span(0,5)) = intersectPoint;

			this->matches3D2D(count, 6) = match2Data[2*i];
			this->matches3D2D(count, 7) = match2Data[2*i+1];

			count++;
		}
	}
	timer.stop();
	cout << "Number of 3D-2D matches: " << count << endl;
	cout << "Time to match 3D-2D: " << timer.getElapsedTimeInMilliSec() << " ms" << endl;

	if (count > 0)
		return this->matches3D2D.rows(0, count-1);
	else
		return arma::mat(0, this->matches3D2D.n_cols);
}

bool KeypointMatcher3D2D::find3DPointOnMesh(const cv::Point2d& refPoint, rowvec& intersectionPoint) const
{
	bool found = false;
	vec source = zeros<vec>(3);						// Camera center in camera coordinate

	vec homorefPoint;								// Homogeneous reference image point
	homorefPoint << refPoint.x << refPoint.y << 1;

	vec destination = solve(camCamera.GetA(), homorefPoint);

	const umat& facets		 = referenceMesh.GetFacets();			// each row: [vid1 vid2 vid3]
	const  mat& vertexCoords = referenceMesh.GetVertexCoords();		// each row: [x y z]

	double minDepth = INFINITY;

	int nFacets = referenceMesh.GetNFacets();
	for (int i = 0; i < nFacets; i++)
	{
		const urowvec &aFacet	= facets.row(i);					// [vid1 vid2 vid3]
		const mat	  &vABC		= vertexCoords.rows(aFacet);		// 3 rows of [x y z]

		vec bary = KeypointMatcher3D2D::findIntersectionRayTriangle(source, destination, vABC);
		if (bary(0) >= 0 && bary(1) >= 0 && bary(2) >= 0)
		{
			double depth = bary(0)*vABC(0,2) + bary(1)*vABC(1,2) + bary(2)*vABC(2,2);
			if (depth < minDepth)
			{
				minDepth = depth;
				intersectionPoint << aFacet(0) << aFacet(1) << aFacet(2)
								  << bary(0) << bary(1) << bary(2) << endr;

				found = true;
			}
		}
	}

	return found;
}

vec KeypointMatcher3D2D::findIntersectionRayTriangle(const vec& source, const vec& destination, const mat& vABC)
{
	vec direction = destination - source;
	
	mat A = join_rows(vABC.t(), -direction);					// A = [vABC' -d]
	A	  = join_cols(A, join_rows(ones(1, 3), zeros(1,1)));	// A = [A; [1 1 1 0]]

	vec b = join_cols(source, ones(1,1));						// b = [source; 1]

	vec X = solve(A, b);										// X = A \ b

	return X.subvec(0,2);
}
