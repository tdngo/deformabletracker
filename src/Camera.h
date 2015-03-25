//////////////////////////////////////////////////////////////////////////
// Author		:	Ngo Tien Dat
// Email		:	dat.ngo@epfl.ch
// Organization	:	EPFL
// Purpose		:	Represent camera object and its operations
// Date			:	19 March 2012
//////////////////////////////////////////////////////////////////////////

#pragma once

#include <iostream>
#include <armadillo>

class Camera
{

private: 
	arma::mat A;		// Intrinsic matrix  3x3
	arma::mat Rt;		// Extrinsic matrix  3x4
	arma::mat ARt;		// Projection matrix 3x4

public:

	// Default constructor
	Camera() {}

	// Constructor that create a camera in camera coordinate. Rt = [I|0]
	Camera(const arma::mat& A)
	{
		this->A = A;
		Rt << 1 << 0 << 0 << 0 << arma::endr
		   << 0 << 1 << 0 << 0 << arma::endr
		   << 0 << 0 << 1 << 0 << arma::endr;

		ARt = A * Rt;
	}

	// Constructor that create a camera given A and Rt
	Camera(const arma::mat& A, const arma::mat& Rt)
	{
		this->A  = A;
		this->Rt = Rt;

		ARt = A * Rt;
	}

	// Load from intrinsic and extrinsic matrix files
	void LoadFromFile(std::string camIntrFile, std::string camExtFile)
	{
		A. load(camIntrFile);
		Rt.load(camExtFile);

		ARt = A * Rt;
	}

	// Load from projection matrix file
	void LoadFromFile(std::string camProjFile)
	{
		ARt.load(camProjFile);

		// TODO: Compute A, Rt using QR decomposition
	}

	// Get intrinsic matrix A
	const arma::mat& GetA() const {
		return A;
	}

	// Get extrinsic matrix Rt
	const arma::mat& GetRt()  const {
		return Rt;
	}

	// Get projection matrix ART
	const arma::mat& GetARt() const {
		return ARt;
	}

	// Project a list of 3D points onto the image plane
	// Input:
	//  + 3D points: size of #points * 3
	// Output:
	//  + 2D points: size of #points * 2
	arma::mat ProjectPoints(const arma::mat& points3D) const;

	// Project a 3D point onto the image plane
	// Input:
	//  + A 3D point: 3*1
	// Output:
	//  + A 2D point: vector of size 2*1
	arma::vec ProjectAPoint(arma::vec point3D) const;

};

