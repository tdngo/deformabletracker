//////////////////////////////////////////////////////////////////////////
// Author		:	Ngo Tien Dat
// Email		:	dat.ngo@epfl.ch
// Organization	:	EPFL
// Purpose		:	Linear algebra library
// Date			:	15 March 2012
//////////////////////////////////////////////////////////////////////////

#include "LinearAlgebraUtils.h"
#include "armadillo"

using namespace arma;

mat LinearAlgebraUtils::SolveWithConstraints ( const mat& A, const mat& B, const mat& Ac, const mat& Bc )
{
	mat P, Q;
	makeLinearParam(Ac, Bc, P, Q);		// Given Ac * X = Bc, we represent X = PY + Q

	// Solution of the least square problem
	return P * solve(A*P, B-A*Q) + Q;
}

void LinearAlgebraUtils::makeLinearParam( const mat& A, const mat& B, mat& P, mat& Q )
{
	int m = A.n_rows;
	int n = A.n_cols;

	Q = solve(A, B);

	// P is an orthonormal basis for the null space of A
	mat U, V;
	vec s;
	svd(U, s, V, A);

	double tolerance = std::max(m, n) * max(s) * math::eps();

	uvec c = s > tolerance;
	int nR = sum(c);

	if (nR <= n-1){
		P = V.cols(nR, n-1);
	} else {
		P.set_size(V.n_rows, 0);
	}

}

mat LinearAlgebraUtils::PseudoInverse( const mat& A, double lambda )
{
	int m = A.n_rows;
	int n = A.n_cols;

	if (m > n)
	{
		if (lambda > 0)
		{
			return solve(A.t()*A + lambda*eye(n,n), A.t());		// solve(A, B) = A \ B
		}
		else
		{
			return solve(A.t()*A, A.t());
		}
	} 
	else
	{
		if (lambda > 0)
		{
			// We want to compute this: A' / ( A*A' + lambda*eye(n,n) )
			// Using relation: A / B = (B' \ A')'
			mat B = A*A.t() + lambda*eye(m,m);		// B' = B, since B is symmetric
			mat C = solve(B, A);
			
			return C.t();
		} 
		else
		{
			// We want to compute: A' / (A*A')
			return solve(A*A.t(), A);
		}
	}
}

vec LinearAlgebraUtils::LeastSquareSolve( const mat& A, const vec& b, double lambda ) 
{
	if (lambda > 0)
	{
		int m = A.n_rows;
		int n = A.n_cols;

		if (m > n)			// Over-constrained system
		{
			mat AtA = A.t() * A + lambda * eye(n, n);
			mat Atb = A.t() * b;

			return solve(AtA, Atb);
		}
		else if (m < n)		// Under-constrained system
		{
			mat AAt = A * A.t() + lambda * eye(m, m);
			vec y	= solve(AAt, b);			
			
			return A.t() * y;
		}
		else				// As many unknowns as equations
		{
			mat AlamdaI = A + lambda * eye(m, m);
			return solve(AlamdaI, b);
		}
	}
	else 
	{
		return solve(A, b);
	}
}
