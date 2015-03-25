//////////////////////////////////////////////////////////////////////////
// Author		:	Ngo Tien Dat
// Email		:	dat.ngo@epfl.ch
// Organization	:	EPFL
// Purpose		:	3D mesh unconstrained reconstruction
// Date			:	15 March 2012
//////////////////////////////////////////////////////////////////////////

#include "Reconstruction.h"
#include <timer/Timer.h>
#include <Linear/ObjectiveFunction.h>
#include <Linear/EqualConstrFunction.h>
#include <Linear/IneqConstrFunction.h>

using namespace arma;

const double Reconstruction::ROBUST_SCALE = 2;

void Reconstruction::buildCorrespondenceMatrix( const mat& matches )
{
	int nMatches	= matches.n_rows;
	int nVertices	= this->refMesh.GetNVertices();

	this->Minit = zeros(2*nMatches, 3*nVertices);
	const mat& A = this->camCamera.GetA();

	for (int i = 0; i < nMatches; i++)
	{
		const rowvec& vid = matches(i, span(0,2));		// Vertex ids in reference image
		const rowvec& bcs = matches(i, span(3,5));		// Barycentric coordinates in reference image
		const rowvec& uvs = matches(i, span(6,7));		// Image coordinates in input image

		// Vertex coordinates are ordered to be [x1,...,xN, y1,...,yN, z1,...,zN]
		for (int k = 0; k <= 2; k++)
		{
			// First row
			Minit(2*i, vid(0) + k*nVertices) = bcs(0) * ( A(0,k) - uvs(0) * A(2,k) );
			Minit(2*i, vid(1) + k*nVertices) = bcs(1) * ( A(0,k) - uvs(0) * A(2,k) );
			Minit(2*i, vid(2) + k*nVertices) = bcs(2) * ( A(0,k) - uvs(0) * A(2,k) );

			// Second row
			Minit(2*i+1, vid(0) + k*nVertices) = bcs(0) * ( A(1,k) - uvs(1) * A(2,k) );
			Minit(2*i+1, vid(1) + k*nVertices) = bcs(1) * ( A(1,k) - uvs(1) * A(2,k) );
			Minit(2*i+1, vid(2) + k*nVertices) = bcs(2) * ( A(1,k) - uvs(1) * A(2,k) );			
		}
	}
}

vec Reconstruction::computeReprojectionErrors( const TriangleMesh& trigMesh, const mat& matchesInit, const uvec& currentMatchIdxs )
{
	const mat& vertexCoords = trigMesh.GetVertexCoords();
	int		   nMatches		= currentMatchIdxs.n_rows;

	vec errors(nMatches);		// Errors of all matches

	for (int i = 0; i < nMatches; i++)
	{
		// Facet (3 vertex IDs) that contains the matching point
		int idx  = currentMatchIdxs(i);
		int vId1 = (int)matchesInit(idx, 0);
		int vId2 = (int)matchesInit(idx, 1);
		int vId3 = (int)matchesInit(idx, 2);

		// 3D vertex coordinates
		const rowvec& vertex1Coords = vertexCoords.row(vId1);
		const rowvec& vertex2Coords = vertexCoords.row(vId2);
		const rowvec& vertex3Coords = vertexCoords.row(vId3);

		double bary1 = matchesInit(idx, 3);
		double bary2 = matchesInit(idx, 4);
		double bary3 = matchesInit(idx, 5);

		// 3D feature point
		rowvec point3D = bary1*vertex1Coords + bary2*vertex2Coords + bary3*vertex3Coords;
		
		// TODO: Implement this function in stead of call projecting function for a single point. This can save expense of function calls
		// Projection
		vec point2D = camCamera.ProjectAPoint(point3D.t());
			
		vec matchingPoint(2);
		matchingPoint(0) = matchesInit(idx, 6);
		matchingPoint(1) = matchesInit(idx, 7);

		errors(i) = norm(point2D - matchingPoint, 2);
	}

	return errors;
}

void Reconstruction::reconstructPlanarUnconstr( const uvec& matchIdxs, double wr, LaplacianMesh& resMesh )
{
	Timer timer;
	
	const mat&	paramMat = this->refMesh.GetParamMatrix();	// Parameterization matrix		
	
	// Build the matrix MPwAP = [MP; wr*AP] and compute: (MPwAP)' * (MPwAP)
	this->computeCurrentMatrices( matchIdxs, wr);

	// --------------- Eigen value decomposition --------------------------
	mat V;
	vec s;
	timer.start();
	eig_sym(s, V, this->MPwAPtMPwAP);
	timer.stop();
	//cout << "Eigen(): " << timer.getElapsedTimeInMilliSec() << " ms"<< endl;
	const vec& c = V.col(0);
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	mat matC = reshape(c, refMesh.GetNCtrlPoints(), 3);

	// Update vertex coordinates
	resMesh.SetVertexCoords(paramMat * matC);
	
	// Resulting mesh yields a correct projection on image but it does not preserve lengths. 
	// So we need to compute the scale factor and multiply with matC

	// Determine on which side the mesh lies: -Z or Z
	double meanZ = mean(resMesh.GetVertexCoords().col(2));
	int globalSign = meanZ > 0 ? 1 : -1;

	const vec& resMeshEdgeLens = resMesh.ComputeEdgeLengths();
	const vec& refMeshEdgeLens = refMesh.GetEdgeLengths();

	double scale = globalSign * norm(refMeshEdgeLens, 2) / norm(resMeshEdgeLens, 2);

	// Update vertex coordinates
	resMesh.SetVertexCoords(scale * paramMat * matC);
}

void Reconstruction::ReconstructPlanarUnconstrIter( const mat& matchesInit, LaplacianMesh& resMesh, uvec& inlierMatchIdxs )
{
	// Input check
	if (matchesInit.n_rows == 0) {
		inlierMatchIdxs.resize(0);
		return;
	}

	Timer timer;

	double	wr		= this->wrInit;			// Currently used regularization weight 
	double	radius	= this->radiusInit;		// Currently used radius of the estimator
	vec		reprojErrors;					// Reprojection errors

	// First, we need to build the correspondent matrix with all given matches to avoid re-computation
	this->buildCorrespondenceMatrix(matchesInit);

	// Then compute MPinit. Function reconstructPlanarUnconstr() will use part of MPinit w.r.t currently used matches
	this->MPinit = this->Minit * this->refMesh.GetBigParamMat();

	uvec matchesInitIdxs = linspace<uvec>(0, matchesInit.n_rows-1, matchesInit.n_rows);

	// Currently used matches represented by their indices. Initially, use all matches: [0,1,2..n-1]
	inlierMatchIdxs = matchesInitIdxs;
	
	for (int i = 0; i < nUncstrIters; i++)
	{
		this->reconstructPlanarUnconstr(inlierMatchIdxs, wr, resMesh);
		
		// If it is the final iteration, break and don't update "inlierMatchIdxs" or "weights", "radius"
		if (i == nUncstrIters - 1) {
			//cout << "Current radius: " << radius << endl;
			//cout << "Current wr: " << wr << endl;
			//Reconstruction::computeCurrentMatrices( currentMatchIdxs, 325 );	// For Fern
			break;
		}

		// Otherwise, remove outliers
		int iterTO = nUncstrIters - 2;
		if (i >= iterTO)
			reprojErrors = this->computeReprojectionErrors(resMesh, matchesInit, matchesInitIdxs);
		else
			reprojErrors = this->computeReprojectionErrors(resMesh, matchesInit, inlierMatchIdxs);

		uvec idxs = find( reprojErrors < radius );
		if ( idxs.n_elem == 0 )
			break;

		if (i >= iterTO)
			inlierMatchIdxs = matchesInitIdxs.elem( idxs );
		else
			inlierMatchIdxs = inlierMatchIdxs.elem( idxs );

		// Update parameters
		wr		= wr 	 / Reconstruction::ROBUST_SCALE;
		radius	= radius / Reconstruction::ROBUST_SCALE;
	}
}

void Reconstruction::computeCurrentMatrices( const uvec& matchIdxs, double wr ) 
{
	int	nMatches = matchIdxs.n_rows;			// Number of currently used matches

	// Build matrix currentMP by taking some rows of MP corresponding to currently used match indices
	Timer timer;
	timer.start();
	mat currentMP(2 * nMatches, this->MPinit.n_cols);
	for (int i = 0; i < nMatches; i++)
	{
		currentMP.rows(2*i, 2*i+1) = this->MPinit.rows(2*matchIdxs(i), 2*matchIdxs(i) + 1);
	}
	timer.stop();
	//cout << "Build current MP matrix: " << timer.getElapsedTimeInMilliSec() << " ms"<< endl;

	timer.start();
	MPwAP 		= join_cols( currentMP, wr*refMesh.GetBigAP() );	// TODO: Avoid computing this. Only needed in the last iteration
	MPwAPtMPwAP = currentMP.t() * currentMP + wr*wr * this->APtAP;
	timer.stop();
	//cout << "Build (MPwAP)' * (MPwAP): " << timer.getElapsedTimeInMilliSec() << " ms"<< endl;
}

void Reconstruction::ReconstructEqualConstr( const vec& cInit, LaplacianMesh& resMesh )
{
	const mat& paramMat = refMesh.GetParamMatrix();
	const mat& bigP		= refMesh.GetBigParamMat();
	
	static bool isFirstFrame = true;
	static vec cOptimal = reshape(refMesh.GetVertexCoords().rows(refMesh.GetCtrlPointIDs()), refMesh.GetNCtrlPoints()*3, 1 );

	// Objective function: use MPwAP which was already computed in unconstrained reconstruction
	ObjectiveFunction *objtFunction;

	if ( this->useTemporal && !isFirstFrame ) {
		objtFunction = new ObjectiveFunction( this->GetMPwAP(), this->timeSmoothAlpha, cOptimal );
	} else {
		objtFunction = new ObjectiveFunction( this->GetMPwAP() );
	}

	// Constrained function
 	EqualConstrFunction cstrFunction( bigP, refMesh.GetEdges(), refMesh.GetEdgeLengths() );

 	if ( this->usePrevFrameToInit && !isFirstFrame )
 		cOptimal = equalConstrOptimize.OptimizeNullSpace(cOptimal, *objtFunction, cstrFunction);
 	else
		cOptimal = equalConstrOptimize.OptimizeNullSpace(cInit, *objtFunction, cstrFunction);

 	mat cOptimalMat = reshape(cOptimal, refMesh.GetNCtrlPoints(), 3);
	if ( cOptimalMat(0,2) < 0 ) {		// Change the sign if the reconstruction is behind the camera. This happens because we take cOptimal as initial value for constrained optimization.
		cOptimalMat = -cOptimalMat;
	}

	// Update vertex coordinates
	resMesh.SetVertexCoords(paramMat*cOptimalMat);

	isFirstFrame = false;
	delete objtFunction;
}

void Reconstruction::ReconstructIneqConstr( const vec& cInit, LaplacianMesh& resMesh )
{
	const mat& paramMat = refMesh.GetParamMatrix();
	const mat& bigP		= refMesh.GetBigParamMat();

	static bool isFirstFrame = true;
	static vec cOptimal = reshape(refMesh.GetVertexCoords().rows(refMesh.GetCtrlPointIDs()), refMesh.GetNCtrlPoints()*3, 1 );

	// Objective function: use MPwAP which was already computed in unconstrained reconstruction
	ObjectiveFunction *objtFunction;

	if ( this->useTemporal && !isFirstFrame  ) {
		objtFunction = new ObjectiveFunction( this->GetMPwAP(), this->timeSmoothAlpha, cOptimal );
	} else {
		objtFunction = new ObjectiveFunction( this->GetMPwAP() );
	}

	// Constrained function
	IneqConstrFunction cstrFunction( bigP, refMesh.GetEdges(), refMesh.GetEdgeLengths() );

 	if ( this->usePrevFrameToInit && !isFirstFrame )
 		cOptimal = ineqConstrOptimize.OptimizeLagrange(cOptimal, *objtFunction, cstrFunction);
 	else
		cOptimal = ineqConstrOptimize.OptimizeLagrange(cInit, *objtFunction, cstrFunction);

	mat cOptimalMat = reshape(cOptimal, refMesh.GetNCtrlPoints(), 3);
	if ( cOptimalMat(0,2) < 0 ) {		// Change the sign if the reconstruction is behind the camera. This happens because we take cOptimal as initial value for constrained optimization.
		cOptimalMat = -cOptimalMat;
	}

	// Update vertex coordinates
	resMesh.SetVertexCoords(paramMat*cOptimalMat);

	isFirstFrame = false;
	delete objtFunction;
}





