//////////////////////////////////////////////////////////////////////////
// Author		:	Ngo Tien Dat
// Email		:	dat.ngo@epfl.ch
// Organization	:	EPFL
// Purpose		:	Triangle mesh class
// Date			:	12 March 2012
//////////////////////////////////////////////////////////////////////////

#include "TriangleMesh.h"
#include <set>

using namespace std;
using namespace arma;

class Edge {
public:
	Edge(unsigned int a, unsigned int b) {
		v1 = min(a,b);		// v1 is always smaller than v2
		v2 = max(a,b);
	}
	unsigned int v1;
	unsigned int v2;
};

class Compare {
public:
	bool operator()(Edge e1, Edge e2) {		// Implement < comparision
		if ( (e1.v1 < e2.v1) || ((e1.v1 == e2.v1) && (e1.v2 < e2.v2)) )
			return true;
		else
			return false;
	}
};

TriangleMesh::TriangleMesh(arma::mat coords, arma::umat faces)
{
	this->vertexCoords = coords;
	this->facets	   = faces;

	// Compute independent attributes
	this->computeEdges();
	this->ComputeEdgeLengths();
	this->computeFacetNormalsNCentroids();
	this->XYZColumn = reshape(this->vertexCoords, GetNVertices()*3, 1);

	this->computeFacetPairs();
}

void TriangleMesh::Load(string baseFile)
{
	// --------- Loading data -----------
	string ptsFile = baseFile + ".pts";
	string triFile = baseFile + ".tri";

	this->vertexCoords.load(ptsFile);
	this->facets.load(triFile);

	// Compute independent attributes
	this->computeEdges();
	this->ComputeEdgeLengths();
	this->computeFacetNormalsNCentroids();
	this->XYZColumn = reshape(this->vertexCoords, GetNVertices()*3, 1);
	this->computeFacetPairs();
}

void TriangleMesh::computeEdges() 
{
	int count = 0;
	this->edges.set_size(2,GetNFacets() * 3);
	umat anEdge(2,1);				// An edge represented as a column vector of vertex ids

	set<Edge,Compare> itemSet;		// To check if an edge is already counted

	int nFacets	  = this->GetNFacets();
	for (int i = 0; i < nFacets; i++)
	{
		if (itemSet.insert(Edge(facets(i,0),facets(i,1))).second) {
			anEdge << facets(i,0) << endr << facets(i,1) << endr;
			this->edges.col(count++) = anEdge;
		}

		if (itemSet.insert(Edge(facets(i,0),facets(i,2))).second) {
			anEdge << facets(i,0) << endr << facets(i,2) << endr;
			this->edges.col(count++) = anEdge;
		}

		if (itemSet.insert(Edge(facets(i,1),facets(i,0))).second) {
			anEdge << facets(i,1) << endr << facets(i,0) << endr;
			this->edges.col(count++) = anEdge;
		}

		if (itemSet.insert(Edge(facets(i,1),facets(i,2))).second) {
			anEdge << facets(i,1) << endr << facets(i,2) << endr;
			this->edges.col(count++) = anEdge;
		}

		if (itemSet.insert(Edge(facets(i,2),facets(i,0))).second) {
			anEdge << facets(i,2) << endr << facets(i,0) << endr;
			this->edges.col(count++) = anEdge;
		}

		if (itemSet.insert(Edge(facets(i,2),facets(i,1))).second) {
			anEdge << facets(i,2) << endr << facets(i,1) << endr;
			this->edges.col(count++) = anEdge;
		}
	}

	this->edges.resize(2,count);
}

SpMat<char> TriangleMesh::computeVertexNeighborhoods()
{
	int nVertices = this->GetNVertices();
	int nFacets	  = this->GetNFacets();

	umat locations(2,nFacets*6);
	Col<char>  values = ones< Col<char> >(nFacets*6);

	urowvec rowIds(6), colIds(6);

	for (int i = 0; i < nFacets; i++)
	{
//		neighborhoods(facets(i,0), facets(i,1)) = 1;
//		neighborhoods(facets(i,0), facets(i,2)) = 1;
//		neighborhoods(facets(i,1), facets(i,0)) = 1;
//		neighborhoods(facets(i,1), facets(i,2)) = 1;
//		neighborhoods(facets(i,2), facets(i,0)) = 1;
//		neighborhoods(facets(i,2), facets(i,1)) = 1;

		rowIds << facets(i,0) << facets(i,0) << facets(i,1) << facets(i,1) << facets(i,2) << facets(i,2) << endr;
		colIds << facets(i,1) << facets(i,2) << facets(i,0) << facets(i,2) << facets(i,0) << facets(i,1) << endr;
		locations(0, span(6*i, 6*i+5)) = rowIds;
		locations(1, span(6*i, 6*i+5)) = colIds;
	}

	SpMat<char> neighborhoods(locations, values, nVertices, nVertices);

	return neighborhoods;
}

const vec& TriangleMesh::ComputeEdgeLengths() 
{
	int nEdges = this->GetNEdges();		// Number of edges

	this->edgeLengths = vec(nEdges);

	for (int i = 0; i < nEdges; i++)
	{
		int i1 = this->edges(0,i);		// Id of the first vertex
		int i2 = this->edges(1,i);		// Id of the second vertex

		const rowvec& c1 = vertexCoords.row(i1);
		const rowvec& c2 = vertexCoords.row(i2);

		this->edgeLengths(i) = arma::norm(c1-c2, 2);
	}

	return this->edgeLengths;
}

void TriangleMesh::computeFacetNormalsNCentroids()
{
	int nFacets = this->GetNFacets();
	this->facetNormals.resize(nFacets, 3);
	this->facetNormalMags.resize(nFacets);
	this->facetCentroids.resize(nFacets, 3);

	for (int i = 0; i < nFacets; i++)
	{
		int ia = this->facets(i,0);
		int ib = this->facets(i,1);
		int ic = this->facets(i,2);

		const rowvec& a = this->vertexCoords.row(ia);
		const rowvec& b = this->vertexCoords.row(ib);
		const rowvec& c = this->vertexCoords.row(ic);

		// Centroid
		this->facetCentroids.row(i) = (a + b + c) / 3;

		// Normal
		rowvec ab = a - b;
		rowvec ac = a - c;
		rowvec normVec	 = cross(ab, ac);
		double magnitude = norm(normVec, 2);

		normVec = normVec / magnitude;

		this->facetNormalMags(i)  = magnitude;
		this->facetNormals.row(i) = normVec;
	}
}

void TriangleMesh::computeFacetPairs()
{
	// TODO: This is a simple algorithm. We optimize later

	int nFacePairs = 0;
	this->facetPairs.set_size(GetNEdges(),2);

	urowvec aPair(1,2);
	
	// Iterate through all facets and check of two of them have a common edge
	int nFacets	  = this->GetNFacets();
	for (int i = 0; i < nFacets; i++)
	{
		for (int j = 0; j < i; j++)
		{
			const urowvec& abc = this->facets.row(i);
			const urowvec& def = this->facets.row(j);
			
			int a = abc(0);
			int b = abc(1);
			int c = abc(2);

			int d = def(0);
			int e = def(1);
			int f = def(2);

			// Check if two facets abc and def have a common edges
			int count = 0;
			if (a == d || a == e || a == f)
			{
				count++;
			}

			if (b == d || b == e || b == f)
			{
				count++;
			}

			if (c == d || c == e || c == f)
			{
				count++;
			}

			// Share two vertices -> share an edge
			if (count == 2) 
			{
				aPair << j << i << endr;
				this->facetPairs.row(nFacePairs++) = aPair;
			}
		}
	}

	this->facetPairs.resize(nFacePairs, 2);

	// Sort pairs so that we can compare to pairs generated by Matlab
	// Sort according to the first column, then second column
	// Can be remove after final checking if all things are correct
	// NOT USED ANY MORE
	/*uvec indices = sort_index(facetPairs.col(0));
	umat sortedPairs(facetPairs.n_rows, facetPairs.n_cols);

	for (int i = 0; i < indices.n_elem; i++)
	{
		sortedPairs.row(i) = facetPairs.row(indices(i));
		
		if (i > 0 && sortedPairs(i,0) == sortedPairs(i-1,0) && sortedPairs(i,1) < sortedPairs(i-1,1))		// If the values in first column are equal, using next column to sort. We can do so since there is at most two equal values in the first column
		{
			sortedPairs.row(i)	 = facetPairs.row(indices(i-1));
			sortedPairs.row(i-1) = facetPairs.row(indices(i));
		}
	}

	this->facetPairs = sortedPairs;*/
}

void TriangleMesh::SetVertexCoords(const mat& vtCoords) {
	this->vertexCoords = vtCoords;

	// Have to update (reset) dependent variables. 
	// Facets, edges, facet pairs information do not change
	this->edgeLengths.reset();
	this->XYZColumn.reset();
	this->facetNormals.reset();
	this->facetNormalMags.reset();
	this->facetCentroids.reset();
	this->vertexNormals.reset();
}

void TriangleMesh::TransformToCameraCoord( const Camera& worldCamera )
{
	// Homogeneous coords of vertices. Insert 1 at the end: each row is [x y z 1]
	mat homoVertexCoords = join_rows(vertexCoords, ones<vec>(vertexCoords.n_rows));

	// Vertices in CAMERA coordinate system
	mat newHomoVertexCoords = worldCamera.GetRt() * homoVertexCoords.t();

	this->SetVertexCoords(newHomoVertexCoords.t());
	this->ComputeEdgeLengths();
}
