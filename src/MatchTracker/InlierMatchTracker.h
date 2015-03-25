/*
 * InlierMatchTracker.h
 *
 *  Created on: Oct 10, 2012
 *      Author: tdngo, datngotien@gmail.com
 */

#ifndef INLIERMATCHTRACKER_H_
#define INLIERMATCHTRACKER_H_

#include <opencv/cv.h>
#include <armadillo>
#include "LKPointTracker.h"

class InlierMatchTracker
{
public:
	InlierMatchTracker();

	// Track inlier matches found in the previous frame
	// INPUT:
	//		@ currentFrameGray	: Frame being processed
	//		@ inlierMatches		: Inlier matches found in previous frame
	// OUPUT:
	//		@ Return those matches that are tracked
	arma::mat TrackMatches(const cv::Mat& currentFrameGray, const arma::mat& inlierMatches);

	// Get image points of tracked matches
	const std::vector<cv::Point2f>& GetGoodTrackedPoints();

private:
	LKPointTracker 	pointsTracker;

	std::vector<cv::Point2f> pointsToTrack, trackedPoints;	// Points to be tracked and tracking results
	std::vector<uchar> 	status;								// Indicate if a point is tracked successfully

	std::vector<bool> 	chosen;								// This variable is used to remove duplicate matches

	arma::mat 		trackedMatches;							// To store results of match tracking
	std::vector<cv::Point2f> goodTrackedPoints;				// Image points of tracked matches
};

#endif /* INLIERMATCHTRACKER_H_ */
