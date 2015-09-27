/*
 * LKPointTracker.cpp
 *
 *  Created on: Jul 12, 2012
 *      Author: tdngo, datngotien@gmail.com
 */

#include "LKPointTracker.h"
#include <opencv/highgui.h>

using namespace std;
using namespace cv;

LKPointTracker::LKPointTracker()
{
  termcrit      = TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
  subPixWinSize = Size(10, 10);
  winSize       = Size(31, 31);
}

void LKPointTracker::TrackPoints(const Mat& currentFrameGray, const vector<Point2f> &pointsToTrack, vector<Point2f> &trackedPoints, vector<uchar> &status)
{
  assert ( currentFrameGray.channels() == 1 );

  // If it is the first frame of the sequence, no tracking is possible
  if( this->previousFrameGray.rows == 0 || pointsToTrack.size() == 0)
  {
    trackedPoints   = vector<Point2f>(0);
    status       = vector<uchar  >(0);
  }
  else
  {
    vector<float> err;
    calcOpticalFlowPyrLK(this->previousFrameGray, currentFrameGray, pointsToTrack, trackedPoints, status, err, winSize, 3, termcrit, 0, 0.001);
  }

  // Set previous frame to be the input current frame
  this->previousFrameGray = currentFrameGray.clone();
}
