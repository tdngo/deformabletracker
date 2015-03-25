//////////////////////////////////////////////////////////////////////////
// Author   :	Ngo Tien Dat
// Email    :	dat.ngo@epfl.ch
// Organization:	EPFL
// Purpose  :	Class of constrained functions inherited from Function.
//					  Edge lengths of input mesh must be as equal as those of
//					  the reference mesh: EQUALITY constraints
// Date     :	30 March 2012
//////////////////////////////////////////////////////////////////////////

#pragma once

#include "Function.h"

class EqualConstrFunction : public Function
{

private:
	const arma::mat		&P;				    // Matrix P in which x = P*c. Size of 3*#vertices x 3*#controlPoints
	const arma::umat	&edges;			  // Edges, size of 2 * #edges each of which is represented by two vertex ids.
	const arma::vec		&refEdgeLens;	// Edge lengths of the reference mesh, size of #edges x 1
										              // Here, we uses const & to avoid deep object copying and get better performance

public:

	// Constructor that initializes the references. p stands for parameters
	EqualConstrFunction( const arma::mat& pP, const arma::umat& pEdges, const arma::vec& pReferencLens ) :
						 P          ( pP ),
						 edges		  ( pEdges ),
						 refEdgeLens( pReferencLens )
	{
		// Initialize variable F and J so that they can be re-used in Evaluate(x)
		int nEdges		= refEdgeLens.n_elem;
		int nActualVars = this->P.n_cols;

		this->F.set_size(nEdges);
		this->J.set_size(nEdges, nActualVars);
	}

	// Implement the pure virtual function declared in the parent class
	// F and J are updated in this function
	// 
	// This implements EQUALITY constraints, therefore, all edges are taken into account.
	// Input:
	//	@param: x is coordinate of control points in the form x1 x2.. y1 y2..z1 z2
	// Output:
	//  + Update attributes: J and F
	virtual void Evaluate( const arma::vec& x);

};

