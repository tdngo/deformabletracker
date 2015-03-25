/*
 * RealtimeDemo.h
 * For deformable planar surface reconstruction
 *
 *  Created on: Jan 25, 2013
 *      Author: tdngo
 */

#ifndef REALTIMEDEMONSTRATOR_H_
#define REALTIMEDEMONSTRATOR_H_

#include <iostream>
#include <time.h>

#include <armadillo>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <timer/Timer.h>
#include <Mesh/LaplacianMesh.h>
#include <PointMatching/FernKeyPointMatcher3D2D.h>
#include <Utils/DUtils.h>
#include <Utils/Visualization.h>
#include <Utils/DefinedMacros.h>
#include <Reconstruction.h>
#include <Camera.h>
#include <MatchTracker/InlierMatchTracker.h>

class RealtimeDemo {

private:
	static const unsigned int DETECT_THRES;	// Threshold on number of inliers to detect the surface

	Timer 	timer;
	bool	isDisplayPoints;					              // Whether display key points in visualization

	// Path and file names
	string 	modelCamIntrFile;					// Camera used for building the model
	string 	modelCamExtFile;
	string 	trigFile;
	string 	refImgFile;
	string  imCornerFile;
	string  videoFile;							// We can use frames in a video to do reconstruction

	Camera  modelWorldCamera, modelCamCamera;	// Camera used for building the model in world reference and camera reference.

	Camera	webCam;								// Use web camera parameters to do reconstruction. In some cases, a different camera was used to build the model.
	string  webcamIntrFile;

	cv::VideoCapture capture;

	cv::Mat  refImg, inputImg;					// Reference image and input image
	IplImage referenceImgRGB;

protected:
	string 						    dataFolder;				// Path to data folder
	arma::urowvec  				ctrPointIds;			// Set of control points
	LaplacianMesh 				*refMesh, resMesh;		// Select planer or non-planer reference mesh

private:
	FernKeypointMatcher3D2D*	keypointMatcher;
	InlierMatchTracker			  matchTracker;
	Reconstruction* 			    reconstruction;

	arma::uvec					inlierMatchIdxs;
	arma::mat   				matchesAll, matchesInlier;

public:
	// Constructor
	RealtimeDemo();

	// Destructor
	virtual ~RealtimeDemo();

	// Set data folder
	void SetDataFolder(const string& dataFolder);

	// If video file is set, then capture frames from the video
	void SetVideoFile(const string& videoFile);

	// Parent object must call this method to init various attributes of this object
	void Init();

	// Main loop
	void Run();

	Reconstruction* GetRecontructionObject() {
		return this->reconstruction;
	}

	// Set whether display keypoints
	void SetDisplayPoints(bool isDisplayPoints) {
		this->isDisplayPoints = isDisplayPoints;
	}

protected:
	// For planar surface. Used for polimorphism
	virtual void initSharedVariables();

private:

	void loadCamerasAndRefMesh();

	void loadRefImageAndPointMatcher();

	void initCaptureDevice();

	// Visualize recontruction results
	void makeVisualization();
};

#endif /* REALTIMEDEMO_H_ */
