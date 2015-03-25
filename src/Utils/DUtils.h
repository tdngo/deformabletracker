//////////////////////////////////////////////////////////////////////////
// Author		:	Ngo Tien Dat
// Email		:	dat.ngo@epfl.ch
// Organization	:	EPFL
// Purpose		:	Some utility functions
// Date			:	12 March 2012
//////////////////////////////////////////////////////////////////////////

#pragma once

#include <string>
#include <sstream>

namespace DUtils
{
	#define ASSERT(condition, message) \
	do { \
		if (! (condition)) { \
			std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
					  << " line " << __LINE__ << ": " << message << std::endl; \
			std::exit(EXIT_FAILURE); \
		} \
	} while (false)

	// NOTICE: Template function can only be defined in header file.
	// Get string representation of numbers
	// If desiredLenZeroPad <= 1: no zero padding
	// If desiredLenZeroPad  > 1: pad zeros to make the string has length as value stored in desiredLenZeroPad
	template<class T>
	std::string ToString(T t, int desiredLenZeroPad = 0, std::ios_base & (*f)(std::ios_base&) = std::dec)
	{
		std::ostringstream oss;
		oss << f << t;

		// Pad zeros in the front so that the resulting string has "strLen" in length
		std::string result = oss.str();
		int initialLen = result.length();
		if (desiredLenZeroPad > 1)
		{
			for (int i = 0; i < desiredLenZeroPad - initialLen; ++i)
			{
				result = "0" + result;
			}
		}

		return result;
	}

	// Estimates the frame per second of the application
	double FPSCalculation();

	// Get absolute path from relative path
	std::string GetFullPath(std::string relativePath);
}
