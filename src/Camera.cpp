//////////////////////////////////////////////////////////////////////////
// Author		:	Ngo Tien Dat
// Email		:	dat.ngo@epfl.ch
// Organization	:	EPFL
// Purpose		:	Represent camera object and its operations
// Date			:	19 March 2012
//////////////////////////////////////////////////////////////////////////

#include "Camera.h"

using namespace arma;

mat Camera::ProjectPoints(const mat& points3D) const
{
	int nVertices = points3D.n_rows;

	// Projected points
	mat points2D(nVertices, 2);

	for (int i = 0; i < nVertices; i++)
	{		
		vec point3D = points3D.row(i).t();
		points2D.row(i) = ProjectAPoint(point3D).t();
	}

	return points2D;
}

vec Camera::ProjectAPoint(vec point3D) const
{
	// Add homogeneous component
	point3D.resize(4);
	point3D(3) = 1;

	vec pointUVW = ARt * point3D;

	double u = pointUVW(0) / pointUVW(2);
	double v = pointUVW(1) / pointUVW(2);

	vec point2D;
	point2D << u << v;

	return point2D;
}
