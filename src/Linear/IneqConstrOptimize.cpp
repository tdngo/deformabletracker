//////////////////////////////////////////////////////////////////////////
// Author		:	Ngo Tien Dat
// Email		:	dat.ngo@epfl.ch
// Organization	:	EPFL
// Purpose		:	Solve generic INEQUALITY constrained optimization problem
// Date			:	26 March 2012
//////////////////////////////////////////////////////////////////////////

#include "IneqConstrOptimize.h"
#include "LinearAlgebraUtils.h"

using namespace arma;

vec IneqConstrOptimize::OptimizeLagrange(const vec& xInit, Function& objtFunction, Function& cstrFunction)
{
	cstrFunction.Evaluate(xInit);

	vec slacks;								// Initial value for slack variables
	vec x = xInit;						// Variables to be found
	for (int i = 0; i < nIters; i++)
	{
		// Update x and s step by step using LVM algorithm
		IneqConstrOptimize::takeAStepLagrange( x, slacks, objtFunction, cstrFunction );
	}

	return x;
}

vec IneqConstrOptimize::OptimizeNullSpace(const vec& xInit, Function& objtFunction, Function& cstrFunction)
{
	vec x = xInit;						// Variables to be found
	for (int i = 0; i < nIters; i++)
	{
		// Update x step by step
		IneqConstrOptimize::takeAStepNullSpace( x, objtFunction, cstrFunction );
	}

	return x;
}

void IneqConstrOptimize::takeAStepLagrange(vec& x, vec& s, Function& objtFunction, Function& cstrFunction) 
{
	// Evaluate functions
	objtFunction.Evaluate(x);
	cstrFunction.Evaluate(x);

	const vec& F = objtFunction.GetF();		// Function value
	const mat& J = objtFunction.GetJ();		// Jacobian. F and J are reference to function.J and function.F
											// We only need to GetF() and GetJ() once
	const vec& C = cstrFunction.GetF();		// Function value of constraints
	const mat& A = cstrFunction.GetJ();		// Jacobian of constraints

	// Initialize s if not yet initialized
	if (s.n_elem == 0) {
		s = ones(A.n_rows);					// Still don't know why s should be initialized as 1. Value 0 doesn't work -> sensitive parameters???
	}

	// Find dx and ds
	int m = A.n_rows;
	int n = A.n_cols;
	mat R = 2 * diagmat(s);
	mat P = s % s;				// Element-wise multiplication s = s.^2

#if 0
	printf("This value should be very close to zero: max(abs(C(X)+S^2)) = %f \n", (arma::max)((arma::abs)(C+P)));
#endif

	// Calculate some terms
	mat t1	= join_rows(J.t()*J, A.t());
	t1		= join_rows(t1, zeros(n,m));	// t1 = [J'*J A' zeros(n,m)]

	mat t2	= join_rows(A, zeros(m,m));
	t2		= join_rows(t2, R);				// t2 = [A zeros(m,m) R]

	// Use this.mu 							// 1e6 if all edges, 1e8 if using real inequality
	mat t3	  = join_rows(zeros(m,n), R);
	t3		  = join_rows(t3, mu*eye(m,m));	// t3 = [zeros(m,n) R mu*eye(m,m)]

	mat t4	  = join_cols(t1, t2);
	t4		  = join_cols(t4, t3);			// t4 = [t1; t2; t3]

	mat t5	= join_cols(J.t()*F, C+P);
	t5		= join_cols(t5, mu*s);			// t5 = [J'*F C+P mu*s]

	vec dXdS = - solve(t4, t5);				// = -t4 \ t5		--> Might need to add a damping term???
	vec dx 	 = dXdS.subvec(0, n-1);
	vec ds 	 = dXdS.subvec(m+n, dXdS.n_elem-1);

	// Update x and s
	x = x + dx;
	s = s + ds;
}

void IneqConstrOptimize::takeAStepNullSpace(vec& x, Function& objtFunction, Function& cstrFunction)
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

	double lambda = 1e-6;					// Used to compute pseudo-inverse

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
}
