//////////////////////////////////////////////////////////////////////////
// Author		:	Ngo Tien Dat
// Email		:	dat.ngo@epfl.ch
// Organization	:	EPFL
// Purpose		:	Solve generic EQUALITY constrained optimization problem
// Date			:	30 March 2012
//////////////////////////////////////////////////////////////////////////

#include "EqualConstrOptimize.h"

using namespace arma;

vec EqualConstrOptimize::OptimizeNullSpace(const vec& xInit, Function& objtFunction, Function& cstrFunction)
{
	vec x = xInit;						// Variables to be found
	for (int i = 0; i < nIters; i++)
	{
		// Update x step by step
		EqualConstrOptimize::takeAStepNullSpace( x, objtFunction, cstrFunction );
	}

	return x;
}

vec EqualConstrOptimize::OptimizeLagrange(const vec& xInit, Function& objtFunction, Function& cstrFunction)
{
	vec x = xInit;						// Variables to be found
	for (int i = 0; i < nIters; i++)
	{
		// Update x step by step
		EqualConstrOptimize::takeAStepLagrange( x, objtFunction, cstrFunction );
	}

	return x;
}

void EqualConstrOptimize::takeAStepLagrange(vec& x, Function& objtFunction, Function& cstrFunction) 
{
	// Evaluate functions
	objtFunction.Evaluate(x);
	cstrFunction.Evaluate(x);

	const vec& F = objtFunction.GetF();		// Function value
	const mat& J = objtFunction.GetJ();		// Jacobian. F and J are reference to function.J and function.F
											// We only need to GetF() and GetJ() once
	const vec& C = cstrFunction.GetF();		// Function value of constraints
	const mat& A = cstrFunction.GetJ();		// Jacobian of constraints

	// Find dx
	int m = A.n_rows;
	int n = A.n_cols;

	double lambda = 1e-6;

	// Calculate some terms
	mat t1	= join_rows(J.t()*J, A.t());	// t1 = [J'*J A']					// NOTICE: CAN improve here by precomputed J'*J
	mat t2	= join_rows(A, zeros(m,m));		// t2 = [A zeros(m,m)]
	mat t3	= join_cols(t1, t2);			// t3 = [t1; t2]
	mat t4	= join_cols(J.t()*F, C);		// t4 = [J'*F C]

	vec dXLambda = -LinearAlgebraUtils::LeastSquareSolve(t3, t4, lambda);		// Compute: -t3 \ t4
	vec dX		 = dXLambda.subvec(0, n-1);

	// Update x
	x = x + dX;
}

void EqualConstrOptimize::takeAStepNullSpace(vec& x, Function& objtFunction, Function& cstrFunction) 
{
	// Evaluate functions
	objtFunction.Evaluate(x);
	cstrFunction.Evaluate(x);

	const vec& F = objtFunction.GetF();		// Function value
	const mat& J = objtFunction.GetJ();		// Jacobian. F and J are reference to function.J and function.F
											// We only need to GetF() and GetJ() once
	const vec& C = cstrFunction.GetF();		// Function value of constraints
	const mat& A = cstrFunction.GetJ();		// Jacobian of constraints

	// Find dx
	int n = A.n_cols;

	double lambda = 1e-6;

	// Compute pseudo inverse and projector
	mat pinvA = LinearAlgebraUtils::PseudoInverse(A, lambda);
	mat Proj  = eye(n, n) - pinvA * A;

	// Compute projection increment
	vec dX	= -pinvA * C;

	// Compute objective function increment
	mat JP   = J * Proj;
	mat FJdX = F + J * dX;

	vec dZ	= LinearAlgebraUtils::LeastSquareSolve(JP, FJdX, lambda);
	dX		= dX - Proj * dZ;

	// Update x
	x = x + dX;

//	dX.t().print("dX:");
}
