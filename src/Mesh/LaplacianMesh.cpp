//////////////////////////////////////////////////////////////////////////
// Author		:	Ngo Tien Dat
// Email		:	dat.ngo@epfl.ch
// Organization	:	EPFL
// Purpose		:	Laplacian mesh class
// Date			:	13 March 2012
//////////////////////////////////////////////////////////////////////////

#include "LaplacianMesh.h"

using namespace arma;

void LaplacianMesh::SetCtrlPointIDs(urowvec pCtrPointIds) 
{
	this->ctrPointIds = pCtrPointIds;

	// Reset
	this->ctrlVertices.reset();
}

void LaplacianMesh::SetVertexCoords(const mat& vtCoords) 
{
	TriangleMesh::SetVertexCoords(vtCoords);	// Call method of parent class first

	// Reset dependent attributes:
	this->ctrlVertices.reset();
	this->regMat.reset();
	this->paramMat.reset();
	this->bigReglMat.reset();
	this->bigParamMat.reset();
	this->bigAP.reset();	
}

void LaplacianMesh::computeRegMatrix()
{
	// Number of adjacent facet pairs
	int nPairs	  = facetPairs.n_rows;
	int nVertices = this->GetNVertices();

	// Init the regularization matrix
	regMat.zeros(nPairs, nVertices);

	for (int i = 0; i < nPairs; i++)
	{
		// Find the union of vertex ids of two facets. In total, there are four vertices		
		const urowvec& pair = facetPairs.row(i);
		int facetId1 = pair(0);
		int facetId2 = pair(1);

		const urowvec& facet1 = facets.row(facetId1);
		const urowvec& facet2 = facets.row(facetId2);
		const urowvec& orderedVertexIds = getOrderedVertexIds(facet1, facet2);
		
		int nOrderedVertices = orderedVertexIds.n_elem;
		assert(nOrderedVertices == 4);		

		// Vertex coordinates
		mat X(nOrderedVertices, 3);
		for (int j = 0; j < 4; j++)
		{
			X.row(j) = vertexCoords(orderedVertexIds(j),span::all);
		}

		// Compute Y from X
		mat Y = unfoldFacetPair(X);

		// Compute weights from Y
		vec weights = computeWeights(Y);

		// Insert results into regularization matrix A
		for (int j = 0; j < nOrderedVertices; j++)
		{
			regMat(i, orderedVertexIds(j)) = weights(j);
		}
	}

	// Update the bigReglMat
	this->bigReglMat = kron( eye(3,3), regMat );
}

void LaplacianMesh::computeParamMatrix()
{
	// Find matrix Pc so that: Pc * X = C
	mat Pc = computePcMatrix();

	// Consider X in bary-centric coordinate system of #controlPoints C,
	// meaning X is represented as linear combination of all points in C
	// P is parameterization matrix - coordinates of X in bary-centric coordinate system of C: X = P * C

	// Now we need to solve a least square constrained linear system
	// regMat * P = 0 with constraints Pc * P = I, I: identity
	mat I = eye(this->GetNCtrlPoints(), this->GetNCtrlPoints());
	mat Z = zeros(this->regMat.n_rows, this->GetNCtrlPoints());
	this->paramMat = LinearAlgebraUtils::SolveWithConstraints(regMat, Z, Pc, I);

	/// Update the bigParamMat
	this->bigParamMat = kron( eye(3,3), paramMat );
}

mat LaplacianMesh::computePcMatrix() const
{
	int nCtrlPoints = this->GetNCtrlPoints();
	int nVertices	= this->GetNVertices();
	
	mat Pc = zeros(nCtrlPoints, nVertices);

	for (int i = 0; i < nCtrlPoints; i++)
	{
		Pc(i, ctrPointIds(i)) = 1;
	}

	return Pc;
}

void LaplacianMesh::ComputeAPMatrices()
{
	this->computeRegMatrix();
	this->computeParamMatrix();
	this->bigAP = bigReglMat * bigParamMat;
}

void LaplacianMesh::LoadAPMatrices(std::string dataFolder)
{
	if ( this->regMat.load(dataFolder + "/A.txt") && this->paramMat.load(dataFolder + "/P.txt") )
	{
		this->bigReglMat  = kron( eye(3,3), regMat );
		this->bigParamMat = kron( eye(3,3), paramMat );
		this->bigAP = bigReglMat * bigParamMat;
	} else {
		this->computeRegMatrix();
		this->computeParamMatrix();
		this->bigAP = bigReglMat * bigParamMat;

		this->regMat.save(dataFolder + "/A.txt", raw_ascii);
		this->paramMat.save(dataFolder + "/P.txt", raw_ascii);
	}
}

urowvec LaplacianMesh::getOrderedVertexIds(urowvec facet1, urowvec facet2)
{
	urowvec orderedIds(4);

	urowvec isShared1;
	urowvec isShared2;
	isShared1.zeros(3);
	isShared2.zeros(3);

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			if (facet1(i) == facet2(j))
			{
				isShared1(i) = isShared2(j) = 1;
			}
		}
	}

	// First element: non-shared vertex of facet1
	uvec founds = find(isShared1 == 0);
	orderedIds(0) = facet1(founds(0));

	// Last (4th) element: non-shared vertex of facet2
	founds = find(isShared2 == 0);
	orderedIds(3) = facet2(founds(0));

	// Middle elements
	founds = find(isShared1 == 1);
	orderedIds(1) = facet1(founds(0));
	orderedIds(2) = facet1(founds(1));

	return orderedIds;
}

mat LaplacianMesh::unfoldFacetPair(mat X)
{
	mat Y(4, 2);

	double a = norm(X.row(0) - X.row(1), 2);
	double b = norm(X.row(0) - X.row(2), 2);
	double c = norm(X.row(1) - X.row(2), 2);
	double d = norm(X.row(1) - X.row(3), 2);
	double e = norm(X.row(2) - X.row(3), 2);

	rowvec u = ltc(a, b, c, -1);
	Y.row(0) = u;

	Y.row(1) = zeros(1, 2);

	Y(2, 0)	 = c;
	Y(2, 1)	 = 0;

	u = ltc(d, e, c, 1);
	Y.row(3) = u;

	return Y;
}

rowvec LaplacianMesh::ltc( double a, double b, double c, int sign )
{
	double x = (a*a - b*b + c*c) / (2*c);
	double y = sign * sqrt(a*a - x*x);

	rowvec u;
	u << x << y;

	return u;
}

vec LaplacianMesh::computeWeights( mat Y )
{
	vec weights;

	mat A = trans(Y);
	A.resize(4, 4);
	A.row(2).ones();
	A.row(3).zeros();
	A(3, 0)  = 1;

	vec B;
	B << 0 << 0 << 0 << 1;

	weights = solve(A, B, true);
	weights = weights / norm(weights, 2);

	return weights;
}
