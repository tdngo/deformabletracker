//////////////////////////////////////////////////////////////////////////
// Author		:	Ngo Tien Dat
// Email		:	dat.ngo@epfl.ch
// Organization	:	EPFL
// Purpose		:	3D mesh unconstrained reconstruction
// Date			:	15 March 2012
//////////////////////////////////////////////////////////////////////////

#pragma once

#include <armadillo>

#include <Mesh/TriangleMesh.h>
#include <Mesh/LaplacianMesh.h>
#include "Camera.h"
#include <Linear/IneqConstrOptimize.h>
#include <Linear/EqualConstrOptimize.h>

class Reconstruction
{
private:
	arma::mat Minit;			// Correspondence matrix M given all initial matches: MX = 0
	arma::mat MPinit;			// Precomputed matrix MP in the term ||MPc|| + wr * ||APc||
								        // M, MP will be computed w.r.t input matches

	arma::mat MPwAP;			// Matrix [MP; wr*AP]. Stored for later use in constrained reconstruction
	
	arma::mat APtAP;			// Precomputed (AP)'*(AP): only need to be computed once
	arma::mat MPwAPtMPwAP;// Precomputed (MPwAP)' * MPwAP used in eigen value decomposition

	const LaplacianMesh		&refMesh;		  // Reference mesh in CAMERA coordinate
	const Camera			    &camCamera;		// Camera object in camera coordinate, projection matrix P must be A*[I|0]
											                // since everything is in camera coordinate
	double			wrInit;					  // Initial weight of deformation penalty term ||AX||
	double			radiusInit;				// Initial radius of robust estimator
	int				  nUncstrIters;			// Number of iterations for unconstrained reconstruction
	float			  timeSmoothAlpha;	// Temporal consistency weight

	bool			useTemporal;			  // Use temporal consistency or not
	bool			usePrevFrameToInit;	// Use the reconstruction in the previous frame to
											          // initalize the constrained recontruction in the current frame

	IneqConstrOptimize ineqConstrOptimize;	// Due to accumulation of ill-conditioned errors.
											                    // Set this to be 10 to get same reconstruction as Matlab version

 	EqualConstrOptimize equalConstrOptimize;// Differences from Matlab are around 0.005

public:
	static const double		ROBUST_SCALE;	// Each iteration of unconstrained reconstruction: decrease by this factor

public:

	// Constructor: p stand for parameters
	// The reference mesh must be in CAMERA coordinates
	Reconstruction(const LaplacianMesh &pRefMesh, const Camera &pCamCamera, 
					double pWrInit     = 8400, double pRadiusInit = 80, int	pNUnstrIters = 5,
					int    pNCstrIters = 5,    float  pTimeSmoothAlpha = 50) :

					refMesh			( pRefMesh ),
					camCamera		( pCamCamera ),
					wrInit			( pWrInit ),
					radiusInit		( pRadiusInit ),
					nUncstrIters	( pNUnstrIters ),
					timeSmoothAlpha ( pTimeSmoothAlpha )
	{
		// Compute (AP)' * (AP) to avoid recomputation
		const arma::mat& bigAP = refMesh.GetBigAP();
		this->APtAP = bigAP.t() * bigAP;

		this->ineqConstrOptimize.SetNIterations(pNCstrIters);
		this->equalConstrOptimize.SetNIterations(pNCstrIters);

		this->usePrevFrameToInit = false;
		this->useTemporal 		 = false;
	}

	void SetNConstrainedIterations( int pNCstrIters) {
		this->ineqConstrOptimize.SetNIterations(pNCstrIters);
		this->equalConstrOptimize.SetNIterations(pNCstrIters);
	}

	void SetTimeSmoothAlpha( int pTimeSmoothAlpha) {
		this->timeSmoothAlpha = pTimeSmoothAlpha;
	}

	void SetUseTemporal( bool useTemporal ) {
		this->useTemporal = useTemporal;
	}

	void SetUsePrevFrameToInit( bool usePrevFrameToInit ) {
		this->usePrevFrameToInit = usePrevFrameToInit;
	}

	void SetWrInit(double pWrInit) {
		this->wrInit = pWrInit;
	}

	// Set mu value for inequality reconstruction
	void SetMu(double mu) {
		this->ineqConstrOptimize.SetMuValue(mu);
	}

	// Get precomputed current MPwAP
	const arma::mat& GetMPwAP() const {
		return this->MPwAP;
	}

	// Iterative reconstruction of planar surface. Outliers are gradually removed
	// Using the function reconstructPlanarUnconstr() multiple times
	// Input: 
	//	+ matches between reference and input images, size of #matches x 9:
	//		- matches(:,0:2): vertex ids of 3 vertices. Ids start from 0
	//		- matches(:,3:5): bary-centric coordinates of the feature points
	//		- matches(:,6:7): 2D coordinate of feature points on the input image
	// Output:
	//  + Update the vertex coordinates of the reference argument Laplacian mesh
	//  + Update the reference argument that contains inliers in given matches
	void ReconstructPlanarUnconstrIter( const arma::mat& matches, LaplacianMesh& resMesh, arma::uvec& inlierMatchIdxs );

	// Reconstruction with equality constrained of all the edges
	// Input: 
	//	+ cInit: initial value of variables
	//	+ MPwAP that was already computed in unconstrained reconstruction
	//	+ Reference Laplacian mesh (one attribute of this class)
	// Output:
	//  + Update the vertex coordinates of the reference argument Laplacian mesh
	void ReconstructEqualConstr(const arma::vec& cInit, LaplacianMesh& resMesh);

	// Reconstruction with inequality constrained of all the edges
	// Input: 
	//	+ cInit: initial value of variables
	//	+ MPwAP that was already computed in unconstrained reconstruction
	//	+ Reference Laplacian mesh (one attribute of this class)
	// Output:
	//  + Update the vertex coordinates of the reference argument Laplacian mesh
	void ReconstructIneqConstr(const arma::vec& cInit, LaplacianMesh& resMesh);

private:
	// Given matches between feature points on the mesh (defined by their 
	// bary-centric coordinates in a facet to which they belong) and on those 
	// on the input image, return the matrix M and the vector v such 
	//             M X = v
	// when X, the vector obtained by concatenating the mesh coordinates, yields
	// the correct projections.
	//
	// Input:
	//  + matches	: The matches are expressed as a matrix each row of which
	//				  contains 3 vertex ids, 3 barycentric coordinates, and 2 
	//				  image coordinates. 
	//  + nVertices : Number of vertices of the mesh
	//  + ARt		: The projection matrix
	//  
	// Output:
	//  Update class attributes: M and v
	//	+ M		: Correspondence matrix
	//	+ v		: in MX = v 
	// TODO: Can be precomputed and reused later on.
	void buildCorrespondenceMatrix( const arma::mat& matches );

	// Compute the current matrix MP by taking some rows of MPinit corresponding 
	// to currently used match indices.
	// Then, compute MPwAP = [MP; wr*AP]
	// Then, compute (MPwAP)' * (MPwAP)
	void computeCurrentMatrices( const arma::uvec& matchIdxs, double wr );

	// Compute reprojection errors
	// Input:
	//	+ trigMesh: Triangle mesh that contains vertex coordinates
	//	+ matches: matches reference and input image
	//	+ indices of matches which we need to compute the projection errors
	// Output:
	//  + Reprojection errors for all vertices (a column vector)
	arma::vec computeReprojectionErrors( const TriangleMesh& trigMesh, const arma::mat& matches, const arma::uvec& currentMatchIdxs );

	// 3D mesh reconstruction for input image. 
	// Minimization problem: min||MPc|| + wr * ||APc|| w.r.t c
	// Input: 
	//	+ matches represented by their indices in the initial given matches
	//	+ weight of regularization term
	// Output:
	//  + Update the vertex coordinates of the reference argument Laplacian mesh
	void reconstructPlanarUnconstr( const arma::uvec& matchIdxs, double wr, LaplacianMesh& resMesh );
};

