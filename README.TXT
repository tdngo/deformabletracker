ABOUT
===========
This folder contains sample code and data to track and reconstruct a 3D deformable surface in realtime.

It is based on our paper:
TD Ngo, J Ostlund, P Fua. Template-based Monocular 3D Shape Recovery using Laplacian Meshes. Accepted in IEEE Transactions on Pattern Analysis and Machine Intelligence.
J. Östlund, A. Varol, T.D. Ngo and P. Fua. Laplacian Meshes for Monocular 3D Shape Recovery. 
European Conference on Computer Vision, Florence, 2012.

Folder content:
 + data: sampe data for a paper template 
 + src: source code
 + 3rdparty: location to put prerequisite libraries such as "ferns"
 + CMakeList.txt
 + README.TXT
 + LICENSE

Authors:
 + Ngo Tien Dat, dat.ngo@epfl.ch
 + Jonas Ostlund, jonas.ostlund@epfl.ch
 + Pascal Fua, pascal.fua@epfl.ch

INSTALL
===========
Prerequisites: You need the following libraries installed.
 + OpenCV: computer vision library <http://opencv.org>
 + armadillo: linear algebra library <http://arma.sourceforge.net>
 + ferns: classification-based feature point detection and matching <http://cvlab.epfl.ch/software/ferns>. Download ferns and put them in the "3rdparty" folder.

Installation is performed using "cmake". Make sure you have "cmake" in your system. You may want to edit the "CMakeList.txt" file to point to dependent libraries. 

From the directory of this file, type the following commands in the terminal.

$ mkdir build
$ cd build
$ cmake ..
$ make

Cmake will build a library file and an executable file. To run the demo, type

$./deformabletracker

Print the image "data/template_print.jpg" and deform it in front of computer webcam The system will track and reconstruct the 3D shape of the paper. What we will see is a window showing the image captured by the webcam overlaid by the reprojection of the 3D reconstructed triangle mesh of the deformable surface.







