//////////////////////////////////////////////////////////////////////////
// Author		:	Ngo Tien Dat
// Email		:	dat.ngo@epfl.ch
// Organization	:	EPFL
// Purpose		:	Solve generic INEQUALITY constrained optimization problem
// Date			:	26 March 2012
//////////////////////////////////////////////////////////////////////////

#pragma once

#include "Function.h"

class IneqConstrOptimize
{
protected:
	// ---------- Parameters used by optimization algorithm ----------------
	int 			nIters;			// Number of iterations
	double			mu;				// Control how length inequality contraints are satisfied.

public:
	// Constructor
	IneqConstrOptimize (int nIters = 20) 
	{
		this->nIters = nIters;
		this->mu 	 = 2e6;
	}

	virtual ~IneqConstrOptimize() {}

	// Do INEQUALITY constrained optimization using Lagrange method
	// Input:
	//	@param: xInit: a vector representing initial x
	//	@param: objtFunction: objective function to be minimized
	//	@param: cstrFunction: constrained function
	// Output:
	//  @return: vector x that minimizes objective function and
	//			 satisfies the constraints
	virtual arma::vec OptimizeLagrange(const arma::vec& xInit, Function& objtFunction, Function& cstrFunction);

	// Do INEQUALITY constrained optimization using Null Space method
	// Input:
	//	@param: xInit: a vector representing initial x
	//	@param: objtFunction: objective function to be minimized
	//	@param: cstrFunction: constrained function
	// Output:
	//  @return: vector x that minimizes objective function and
	//			 satisfies the constraints
	virtual arma::vec OptimizeNullSpace(const arma::vec& xInit, Function& objtFunction, Function& cstrFunction);

	void SetMuValue(double mu) {
		this->mu = mu;
	}

	void SetNIterations(int nIters) {
		this->nIters = nIters;
	}

protected:
	// Take a step to change x and slack variables using Lagrangian multipliers method
	// Input:
	//	+ x: actual variables
	//	+ s: slack variables
	//	+ objective function
	//	+ constrained function
	// Output:
	//  + x and slack are updated
	void takeAStepLagrange(arma::vec& x, arma::vec& s, Function& objtFunction, Function& cstrFunction);

	// Take a step to change x and slack variables using Null Space method
	// Input:
	//	+ x: actual variables
	//	+ objective function
	//	+ constrained function
	// Output:
	//  + x is updated
	static void takeAStepNullSpace(arma::vec& x, Function& objtFunction, Function& cstrFunction);
};

