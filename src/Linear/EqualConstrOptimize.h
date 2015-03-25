//////////////////////////////////////////////////////////////////////////
// Author		:	Ngo Tien Dat
// Email		:	dat.ngo@epfl.ch
// Organization	:	EPFL
// Purpose		:	Solve generic EQUALITY constrained optimization problem
// Date			:	30 March 2012
//////////////////////////////////////////////////////////////////////////

#pragma once

#include "Function.h"
#include "LinearAlgebraUtils.h"

class EqualConstrOptimize
{

private:
	// ---------- Parameters used by optimization algorithm ----------------
	int nIters;		// Number of iterations

public:
	// Constructor
	EqualConstrOptimize (int nIters = 20) 
	{
		this->nIters = nIters;
	}

	// Do EQUALITY constrained optimization using Null Space method
	// Input:
	//	@param: xInit: a vector representing initial x
	//	@param: objtFunction: objective function to be minimized
	//	@param: cstrFunction: constrained function
	// Output:
	//  @return: vector x that minimizes objective function and
	//			 satisfies the constraints
	arma::vec OptimizeNullSpace(const arma::vec& xInit, Function& objtFunction, Function& cstrFunction);

	// Do EQUALITY constrained optimization using Lagrange method
	// Input:
	//	@param: xInit: a vector representing initial x
	//	@param: objtFunction: objective function to be minimized
	//	@param: cstrFunction: constrained function
	// Output:
	//  @return: vector x that minimizes objective function and
	//			 satisfies the constraints
	arma::vec OptimizeLagrange(const arma::vec& xInit, Function& objtFunction, Function& cstrFunction);

	void SetNIterations(int nIters) {
		this->nIters = nIters;
	}

private:
	// Take a step to change variable x using the method of Lagrange multipliers
	// Input:
	//	+ x: actual variables
	//	+ objective function
	//	+ constrained function
	// Output:
	//  + x and slack are updated
	static void takeAStepLagrange(arma::vec& x, Function& objtFunction, Function& cstrFunction);

	// Take a step to change variable x using the method of minimizing in 
	// the NULL space of the constraints
	// Input:
	//	+ x: actual variables
	//	+ objective function
	//	+ constrained function
	// Output:
	//  + x is updated
	static void takeAStepNullSpace(arma::vec& x, Function& objtFunction, Function& cstrFunction);

};

