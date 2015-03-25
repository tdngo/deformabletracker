//////////////////////////////////////////////////////////////////////////
// Author		:	Ngo Tien Dat
// Email		:	dat.ngo@epfl.ch
// Organization	:	EPFL
// Purpose		:	Abstract class of function in the form f: R^m --> R^n
//					Used to pass function as parameter into an algorithm
// Date			:	26 March 2012
//////////////////////////////////////////////////////////////////////////

#pragma once

#include <armadillo>

class Function
{
protected:
	arma::vec F;			// Function value
	arma::mat J;			// Jacobian of function

public:

	virtual ~Function() {}

	// Evaluate the function at x. 
	// Update function value F and Jacobian J
	// Set this as a pure virtual function so that inherited classes must 
	// implement this function
	virtual void Evaluate(const arma::vec& x) = 0;
	
	// Get function value evaluated by function Evaluate()
	// Return a const & to avoid object copying
	// Usage: const vec& F = function.GetF();
	const arma::vec& GetF() {
		return this->F;
	}

	// Get function Jacobian evaluated by function Evaluate()
	// Return a const & to avoid object copying
	// Usage: const mat& J = function.GetJ();
	const arma::mat& GetJ() {
		return this->J;
	}
};

