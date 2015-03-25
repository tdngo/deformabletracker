/*
 * InlierMatchTracker.cpp
 *
 *  Created on: Oct 10, 2012
 *      Author: tdngo, datngotien@gmail.com
 */

#include "InlierMatchTracker.h"
#include "PointMatching/KeypointMatcher3D2D.h"
#include <iostream>

using namespace std;
using namespace cv;

InlierMatchTracker::InlierMatchTracker()
{
	chosen 			= vector<bool>(KeypointMatcher3D2D::MAXIMUM_NUMBER_OF_MATCHES);
	trackedMatches 	= arma::mat(KeypointMatcher3D2D::MAXIMUM_NUMBER_OF_MATCHES, 9);
}

arma::mat InlierMatchTracker::TrackMatches(const cv::Mat& currentFrameGray, const arma::mat& inlierMatches)
{
	// Extract inlier points to be tracked. Don't take duplicate matches.
	for (unsigned int i = 0; i < chosen.size(); i++)
	{
		chosen[i] = false;
	}

	pointsToTrack.resize( inlierMatches.n_rows );
	vector<int> trackedMatchIdxs(inlierMatches.n_rows);		// Index of 'inlierMatches' corresponding to the point being tracked

	int count = 0;
	for ( unsigned int i = 0; i < inlierMatches.n_rows; i++ )
	{
		// If model point is unknown or model point is already selected -> disregard this matches
		if (inlierMatches(i,8) < 0 || chosen[inlierMatches(i,8)])
			continue;

		chosen[inlierMatches(i,8)] = true;

		pointsToTrack   [count] = Point2f( inlierMatches(i, 6), inlierMatches(i, 7) );
		trackedMatchIdxs[count] = i;

		count++;
	}
	pointsToTrack.resize(count);

	// Track points
	trackedPoints.clear();
	pointsTracker.TrackPoints( currentFrameGray, pointsToTrack, trackedPoints, status );

	// Compute tracked matches
	this->goodTrackedPoints.clear();
	count = 0;
	for ( unsigned int i = 0; i < trackedPoints.size(); i++ ) {
		if (!status[i])
			continue;

		goodTrackedPoints.push_back( trackedPoints[i] );

		trackedMatches.row(count) = inlierMatches.row(trackedMatchIdxs[i]);
		trackedMatches(count, 6)  = trackedPoints[i].x;
		trackedMatches(count, 7)  = trackedPoints[i].y;

		count++;
	}

	if (count > 0)
		return trackedMatches.rows(0, count-1);
	else
		return arma::mat(0, inlierMatches.n_cols);
}

const vector<Point2f>& InlierMatchTracker::GetGoodTrackedPoints()
{
	return this->goodTrackedPoints;
}
