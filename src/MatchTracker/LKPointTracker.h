/*
 * LKPointTracker.h
 *
 *  Created on: Jul 12, 2012
 *      Author: tdngo, datngotien@gmail.com
 */

#pragma once

#include <opencv/cv.h>

class LKPointTracker
{
public:
	// Default constructor
	LKPointTracker();

	// Track feature points in the previous frame
	// @param: currentFrame: frame in which points are tracked
	// @param: status: whether points are tracked or not?
	void TrackPoints(const cv::Mat& currentFrameGray, const std::vector<cv::Point2f> &pointsToTrack, std::vector<cv::Point2f> &trackedPoints, std::vector<uchar> &status);

private:
	cv::TermCriteria termcrit;
	cv::Size subPixWinSize;
	cv::Size winSize;

	cv::Mat previousFrameGray;

};
