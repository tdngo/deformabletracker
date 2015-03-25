/*
 * RealtimeDemonstrator.cpp
 *
 *  Created on: Jan 25, 2013
 *      Author: tdngo
 */

#include "RealtimeDemo.h"

using namespace arma;

const unsigned int RealtimeDemo::DETECT_THRES = 100;

RealtimeDemo::RealtimeDemo()
{
  // Default data folder
  this->SetDataFolder("./data/");
  videoFile  = "";

  isDisplayPoints = true;
  reconstruction  = NULL;
  keypointMatcher = NULL;
  refMesh         = NULL;
}

RealtimeDemo::~RealtimeDemo() {
	delete this->reconstruction;
	delete this->keypointMatcher;
	delete this->refMesh;
}

void RealtimeDemo::SetDataFolder(const string& dataFolder)
{
	this->dataFolder 	= dataFolder + "/";

	modelCamIntrFile	= this->dataFolder + "cam.intr";
	modelCamExtFile		= this->dataFolder + "cam.ext";
	trigFile			    = this->dataFolder + "mesh";
	refImgFile			  = this->dataFolder + "model.png";
	imCornerFile		  = this->dataFolder + "im_corners.txt";
	webcamIntrFile 		= this->dataFolder + "webcam.intr";
}

void RealtimeDemo::SetVideoFile(const string& videoFile)
{
	this->videoFile = videoFile;
}

void RealtimeDemo::Init()
{
	this->initSharedVariables();		// This will call an appropriate function for planar and non-planar surface

	this->loadCamerasAndRefMesh();
	this->loadRefImageAndPointMatcher();
	this->initCaptureDevice();

	this->isDisplayPoints = true;

	int 	nUnConstrIters	= 5;
	int 	radiusInit 		  = 5  * pow(Reconstruction::ROBUST_SCALE, nUnConstrIters-1);
	double 	wrInit	  		= 525 * pow(Reconstruction::ROBUST_SCALE, nUnConstrIters-1);
	this->reconstruction  	= new Reconstruction  ( *refMesh, webCam, wrInit, radiusInit, nUnConstrIters );
	this->reconstruction->SetUseTemporal(true);
	this->reconstruction->SetUsePrevFrameToInit(true);
}

void RealtimeDemo::initSharedVariables()
{
	ctrPointIds.load(this->dataFolder + "ControlPointIDs.txt");

	// Type of Laplacian mesh
	this->refMesh = new LaplacianMesh();
}

void RealtimeDemo::loadCamerasAndRefMesh()
{
	modelWorldCamera.LoadFromFile(modelCamIntrFile, modelCamExtFile);
	modelCamCamera = Camera( modelWorldCamera.GetA() );

	mat webcamK;
	if (webcamK.load(webcamIntrFile))
		this->webCam = Camera(webcamK);
	else
		this->webCam = this->modelCamCamera;

	refMesh->Load(trigFile);
	refMesh->TransformToCameraCoord(modelWorldCamera);		// Convert the mesh into camera coordinate using world camera
	refMesh->SetCtrlPointIDs(ctrPointIds);
	refMesh->ComputeAPMatrices();
	refMesh->computeFacetNormalsNCentroids();

	// Init the recontructed mesh
	resMesh = *refMesh;
}

void RealtimeDemo::loadRefImageAndPointMatcher()
{
	refImg	 		= cv::imread(refImgFile);
	referenceImgRGB = refImg;

	mat imCorners;
	imCorners.load(this->imCornerFile);

	int pad = 0;
	cv::Point topLeft		(imCorners(0,0) - pad, imCorners(0,1) - pad);
	cv::Point topRight		(imCorners(1,0) + pad, imCorners(1,1) - pad);
	cv::Point bottomRight	(imCorners(2,0) + pad, imCorners(2,1) + pad);
	cv::Point bottomLeft	(imCorners(3,0) - pad, imCorners(3,1) + pad);

	// Init point matcher
	this->keypointMatcher = new FernKeypointMatcher3D2D( &referenceImgRGB,
														 topLeft, topRight, bottomRight, bottomLeft,
														 *refMesh, modelCamCamera, refImgFile );

	Visualization::DrawAQuadrangle  ( &referenceImgRGB, topLeft, topRight, bottomRight, bottomLeft, TPL_COLOR );
	Visualization::DrawProjectedMesh( refImg, *refMesh, modelCamCamera, MESH_COLOR );
	imshow("Reference image", refImg);
}

void RealtimeDemo::initCaptureDevice()
{
	if ( this->videoFile.length() == 0 )
		capture = cv::VideoCapture(CV_CAP_ANY);
	else
		capture = cv::VideoCapture( this->videoFile );

	capture.set(CV_CAP_PROP_FRAME_WIDTH , 640);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	assert(capture.isOpened());
}

void RealtimeDemo::Run()
{
  printf("Press any key to start ...\n");
	cvWaitKey();

	cv::Mat 	inputImgGray;
	IplImage 	inputImgRGB;
	while (true)
	{
		cout << "------------------------------------------------------" << endl;
		capture >> inputImg;

		inputImgRGB =  inputImg;
		cvtColor( inputImg, inputImgGray, CV_BGR2GRAY );

		// ======== Matches between reference image and input image =======
		timer.start();

		matchesAll = keypointMatcher->MatchImages3D2D(&inputImgRGB); // TODO: Make the input gray image

		timer.stop();
		cout << "Number of 3D-2D Fern matches: " << matchesAll.n_rows << endl;
		cout << "Total time for matching: " << timer.getElapsedTimeInMilliSec() << " ms" << endl;

		// ================== Track inlier matches ========================
		mat trackedMatches = matchTracker.TrackMatches(inputImgGray, matchesInlier);
		matchesAll 		   = join_cols(matchesAll, trackedMatches);

		// ============== Reconstruction without constraints ==============
		timer.start();
    reconstruction->ReconstructPlanarUnconstrIter( matchesAll, resMesh, inlierMatchIdxs );

		matchesInlier = matchesAll.rows(inlierMatchIdxs);

		timer.stop();
		cout << "Number of inliers in unconstrained recontruction: " << inlierMatchIdxs.n_rows << endl;
		cout << "Unconstrained reconstruction time: " << timer.getElapsedTimeInMilliSec() << " ms" << endl;

		// ============== Constrained Optimization =======================
		if (inlierMatchIdxs.n_rows > DETECT_THRES)
		{
			timer.start();

			vec cInit = reshape( resMesh.GetCtrlVertices(), resMesh.GetNCtrlPoints()*3, 1 );	// x1 x2..y1 y2..z1 z2..

      reconstruction->ReconstructIneqConstr(cInit, resMesh);

			timer.stop();
			cout << "Constrained reconstruction time: " << timer.getElapsedTimeInMilliSec() << " ms \n\n";
		}

		this->makeVisualization();
	}
}

void RealtimeDemo::makeVisualization()
{
	timer.start();

	// Reproject the mesh vertices on the image
	if (inlierMatchIdxs.n_rows > DETECT_THRES)
	{
		Visualization::DrawProjectedMesh( inputImg, resMesh, webCam, MESH_COLOR );

		if (isDisplayPoints) {
		  Visualization::DrawVectorOfPointsAsPlus(inputImg, matchTracker.GetGoodTrackedPoints(), INLIER_KPT_COLOR);
			Visualization::DrawPointsAsDot( inputImg, matchesAll.cols(6,7), KPT_COLOR );				  // All points
			Visualization::DrawPointsAsDot( inputImg, matchesInlier.cols(6,7), INLIER_KPT_COLOR );// Inlier points
		}
	}

	// Calculate FPS
	double 	fps 	= DUtils::FPSCalculation();
	string 	fpsStr 	= "FPS: " +  DUtils::ToString(fps);
	cv::putText(inputImg, fpsStr, cv::Point(10, 30), CV_FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2);

	cv::imshow( "Mesh drawn on image", inputImg );

	timer.stop();
	cout << "Visualization time: " << timer.getElapsedTimeInMilliSec() << " ms \n\n";

	cvWaitKey(10);
}

