//////////////////////////////////////////////////////////////////////////
// Author			  :  Ngo Tien Dat
// Email			  :  dat.ngo@epfl.ch
// Organization	:  EPFL
// Purpose      :  Real-time reconstruction using webcam
// Date         :  18 May 2012
//////////////////////////////////////////////////////////////////////////

#include <RealtimeDemo/RealtimeDemo.h>

using namespace cv;

// Global variables
RealtimeDemo *demoPtr;
int alpha_slider      = 30;     // Control temporal consistency weight in alpha * ||x - xPrev||
int weightAP_slider   = 525;    // Control regularization weight in ||MPc|| + alpha * ||APc||
int nCstrIters_slider = 4;      // Number of iterations in constrained optimization
int dispPoints_slider = 1;      // Display feature points or not

// Callback for trackbar
void on_alpha_trackbar(int, void*)
{
  cout << "alpha_slider = " << alpha_slider << endl;
  demoPtr->GetRecontructionObject()->SetTimeSmoothAlpha(alpha_slider);
}

void on_weightAP_trackbar(int, void*)
{
  cout << "weightAP_slider = " << weightAP_slider << endl;
  demoPtr->GetRecontructionObject()->SetWrInit(weightAP_slider * 16);
}

void on_NCstrIters_trackbar(int, void*)
{
  demoPtr->GetRecontructionObject()->SetNConstrainedIterations(nCstrIters_slider + 1);
  cout << "nCstrIters = " << nCstrIters_slider + 1 << endl;
}

void on_DispPoints_trackbar(int, void*)
{
  demoPtr->SetDisplayPoints(dispPoints_slider);
  cout << "Display feature points = " << dispPoints_slider << endl;
}

void createParametersWindow()
{
  // Create Windows
  int width  = 500;
  int height = 50;
  const char* winName = "Parameters";
  namedWindow(winName, CV_WINDOW_NORMAL);
  resizeWindow(winName, width, height);

  createTrackbar("Regularization: ", winName, &weightAP_slider, 2000, on_weightAP_trackbar);
  createTrackbar("Temporal consistency: ", winName, &alpha_slider, 200, on_alpha_trackbar);
  createTrackbar("NCstrIters: ", winName, &nCstrIters_slider, 20, on_NCstrIters_trackbar);
  createTrackbar("Display points: ", winName, &dispPoints_slider, 1, on_DispPoints_trackbar);
}

int main(int argc, char** argv)
{
  demoPtr = new RealtimeDemo();
  demoPtr->SetDataFolder("../data");
  demoPtr->Init();
  createParametersWindow();
  demoPtr->SetDisplayPoints(false);

  demoPtr->Run();

  cout << "\nPress any key to continue!!!\n";
  cin.ignore(1);

  delete demoPtr;
  return 0;
}




