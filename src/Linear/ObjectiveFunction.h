//////////////////////////////////////////////////////////////////////////
// Author		:	Ngo Tien Dat
// Email		:	dat.ngo@epfl.ch
// Organization	:	EPFL
// Purpose		:	Objective function f: R^m --> R^n
//									   f(x) = A*x - b
//					Used to pass function as parameter into an algorithm
// Date			:	26 March 2012
//////////////////////////////////////////////////////////////////////////

#pragma once

#include "Function.h"

class ObjectiveFunction : public Function
{

private:
	const arma::mat& 	MP;		// Matrix MP that compute projection error MP*x
								          // We use const & to avoid deep object copying

	float 				  alpha;
	arma::vec  	    xPrev;

	bool		useTemporal;	  // Use temporal consistency or not

public:

	// Constructor that initializes the reference variable MP.
	// xPrev is initialized to avoid error messages, it is not used in this function.
	ObjectiveFunction (const arma::mat& MPmat);

	// Constructor that initializes the reference variable MP and
	// take temporal consistency constraints as an objective too
	// MP * x = 0
	// alpha * I * (x - xPrev) = 0
	ObjectiveFunction (const arma::mat& MPmat, const float alpha, const arma::vec& xPrev);

	// Implement the pure virtual function declared in the parent class
	// Update F and J
	// This objective function has the form F = A * x
	virtual void Evaluate(const arma::vec& x );
};






