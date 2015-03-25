//////////////////////////////////////////////////////////////////////////
// Author		:	Ngo Tien Dat
// Email		:	dat.ngo@epfl.ch
// Organization	:	EPFL
// Purpose		:	Class of constrained functions inherited from Function
//					Edge lengths of input mesh must be smaller than those of
//					the reference mesh: INQUALITY constraints
// Date			:	26 March 2012
//////////////////////////////////////////////////////////////////////////

#include "IneqConstrFunction.h"
#include <math.h>

using namespace arma;
using namespace std;

void IneqConstrFunction::Evaluate( const vec& x )
{
	// Vertex coordinates: First convert control points into vertices
	// verCoords has the form of [x1 x2, x3... y1 y2 y3... z1 z2 z3]
	// We will compute Jacobian w.r.t vertex variables first, then we compute
	// Jacobian w.r.t actual variables using composition rules
	vec vertCoords = this->P * x;

	int nVertVars	= this->P.n_rows;				// Number of vertex variables = 3 * #vertices
	int nVerts		= nVertVars / 3;				// Number of vertices
	int nEdges		= this->refEdgeLens.n_elem;		// Number of edges (#constraints)

	// TODO: Make Jv to be attribute of class to avoid re-allocating each evaluation
	// Jacobian w.r.t all vertex coordinates. Then this->J = Jv * P since x = P*c
	mat Jv = zeros(nEdges, nVertVars);

	// Iterate through all edges to compute function value and derivatives	
	for (int i = 0; i < nEdges; i++)
	{
		// An edge has two vertices
		int vertID1 = edges(0, i);
		int vertID2 = edges(1, i);
		
		// Indices of x,y,z coordinates in the vector vertCoords [x1 x2, x3... y1 y2 y3... z1 z2 z3]
		int x1Idx	= vertID1;
		int y1Idx	= x1Idx + nVerts;
		int z1Idx	= y1Idx + nVerts;

		int x2Idx	= vertID2;		
		int y2Idx	= x2Idx + nVerts;		
		int z2Idx	= y2Idx + nVerts;

		// Coordinates of two vertices (x1,y1,z1) & (x2,y2,z2)
		double x1	= vertCoords(x1Idx);		
		double y1	= vertCoords(y1Idx);
		double z1	= vertCoords(z1Idx);

		double x2	= vertCoords(x2Idx);
		double y2	= vertCoords(y2Idx);		
		double z2	= vertCoords(z2Idx);
		
		// Edge length
		double edgeLen = sqrt( pow(x2-x1, 2) + pow(y2-y1, 2) + pow(z2-z1, 2) );
			
		// Compute function value = "edge length" - "reference edge length"
		this->F(i) = edgeLen - refEdgeLens(i);

    // Compute derivatives. Recall the derivative: sqrt'(x) = 1/(2*sqrt(x))
    Jv(i, x1Idx) = (x1-x2) / edgeLen;
    Jv(i, y1Idx) = (y1-y2) / edgeLen;
    Jv(i, z1Idx) = (z1-z2) / edgeLen;

    Jv(i, x2Idx) = (x2-x1) / edgeLen;
    Jv(i, y2Idx) = (y2-y1) / edgeLen;
    Jv(i, z2Idx) = (z2-z1) / edgeLen;
	}

	// Jacobian w.r.t actual variables using composition chain rule
	this->J = Jv * P;

	vec Fabs = arma::abs(this->F);

#if 0
	cout << "Constraint function mean: " << arma::mean(Fabs) << endl;
#endif

}












