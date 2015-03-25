//////////////////////////////////////////////////////////////////////////
// Author		:	Ngo Tien Dat
// Email		:	dat.ngo@epfl.ch
// Organization	:	EPFL
// Purpose		:	Objective function f: R^m --> R^n
//					Used to pass function as parameter into an algorithm
// Date			:	26 March 2012
//////////////////////////////////////////////////////////////////////////

#include "ObjectiveFunction.h"

ObjectiveFunction::ObjectiveFunction (const arma::mat& MPmat) : MP(MPmat), xPrev(arma::vec())
{
  // Initialize variable F and J so that they can be re-used in Evaluate(x)
  this->F.set_size(MP.n_rows);

  // Jacobian of the function F = MP * x is constant
  this->J = MP;

  useTemporal = false;
  alpha       = 0;
}

ObjectiveFunction::ObjectiveFunction (const arma::mat& MPmat, const float alpha, const arma::vec& xPrev) : MP(MPmat), alpha(alpha), xPrev(xPrev)
{
  // Initialize variable F and J so that they can be re-used in Evaluate(x)
  this->F.set_size(MP.n_rows + xPrev.n_elem);

  // Jacobian of the function F = MP * x + alpha * I * (x - xPrev) is constant
  this->J = arma::join_cols( MP, alpha * arma::eye(xPrev.n_elem, xPrev.n_elem) );

  useTemporal = true;
}

void ObjectiveFunction::Evaluate(const arma::vec& x ) {
  if (useTemporal)
    this->F = arma::join_cols( MP * x, alpha * (x - xPrev) );
  else
    this->F = MP * x;

  // Jacobian of this function is constant and is already computed in Constructor:
  //// this->J = MP;
}
