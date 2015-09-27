//////////////////////////////////////////////////////////////////////////
// Author       :  Ngo Tien Dat
// Email        :  dat.ngo@epfl.ch
// Organization :  EPFL
// Purpose      :  Linear algebra library
// Date         :  15 March 2012
//////////////////////////////////////////////////////////////////////////

#pragma once

#include <iostream>
#include <armadillo>

class LinearAlgebraUtils
{
public:

  // Solve in least squares sense A*X = B subject to Ac*X = Bc w.r.t. X.
  // B, Bc should have the same number of columns
  // Return optimal X
  static arma::mat SolveWithConstraints(const arma::mat& A, const arma::mat& B, const arma::mat& Ac, const arma::mat& Bc);

  // Compute pseudo inverse of a matrix
  // Input:
  //  + A be an m x n matrix. 
  //  + lambda is damping term that is useful for ill conditioned matrices. 
  // Output:
  //  + Return the n x m matrix B such that A*B = I if m>n, 
  //  + Return the n x m matrix B such that B*A = I, otherwise
  static arma::mat PseudoInverse(const arma::mat& A, double lambda);


  // Solve a system of linear equations Ax = b in least squares sense
  // Input:
  //  + A
  //  + b
  //  + lambda: damping term
  // Output:
  //  + Optimal x that minimize least squared error
  static arma::vec LeastSquareSolve(const arma::mat& A, const arma::vec& b, double lambda);

private:
  // Given an under-determined linear system AX = B, this function outputs a
  // parameterization of the solutions for this system: X = PY + Q
  // This is useful for implementing linear equality constraints. B, X, Y, Q
  // have the same number of columns, that is, they are not limited to only one column.
  // Note that: AX=B MUST be an under-determined linear systems, otherwise errors (out of bound) in matrix access
  static void makeLinearParam(const arma::mat& A, const arma::mat& B, arma::mat& P, arma::mat& Q);
};

