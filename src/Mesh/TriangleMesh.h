//////////////////////////////////////////////////////////////////////////
// Author    :  Ngo Tien Dat
// Email    :  dat.ngo@epfl.ch
// Organization  :  EPFL
// Purpose    :  Triangle mesh class
// Date      :  12 March 2012
//////////////////////////////////////////////////////////////////////////

#pragma once

#include <iostream>
#include <assert.h>
#include <armadillo>
#include <Camera.h>

class TriangleMesh
{

protected:
  // Uninitialized mat and vec have the size of zero.

  arma::mat   vertexCoords;  // Coordinates of vertices. Size of #vertices * 3. Number 3 means (x,y,z)
  arma::umat  facets;        // Facets of the mesh. Size of #facets * 3. (vertexId1,vertexId2,vertexId3) vertex id starts from 0

  // ----- Extended dependent attributes ------
  arma::umat edges;          // Edges. Size of 2 * #edges. Each represetned by two vertex ids.

  arma::vec  edgeLengths;    // Edge lengths, a column vector
  arma::vec  XYZColumn;      // Vectorized coordinates of vertices in a column vector with the
                             // order [x1,...,xN,y1,...,yN,z1,...,zN]

  arma::mat  facetNormals;    // Facet normals. Size of #facets * 3
  arma::vec  facetNormalMags; // Facet normal magnitudes, a column vector
  arma::mat  facetCentroids;  // Size of #facets * 3
  arma::mat  vertexNormals;   // Size of #vertices * 3
  arma::umat facetPairs;      // pairs of adjacent facets, size of #pairs * 2 (two facet ids)

public:
  // Constructor
  TriangleMesh(void) {}

  TriangleMesh(arma::mat coords, arma::umat faces);

  // Destructor
  virtual ~TriangleMesh(void) {}

  // Load the triangle mesh from file pair ".pts" and ".trig"
  // Input:
  //  + baseFile: root file name without .pts and .trig
  void Load(std::string baseFile);

  // Transform the mesh into camera coordinate
  // Multiply vertex with Rt matrix and call set vertex function
  void TransformToCameraCoord(const Camera& worldCamera);

  // Get facets
  const arma::umat& GetFacets() const
  {
    return facets;
  }

  // Get edges: size of 2 * #edges. Each represetned by two vertex ids.
  const arma::umat& GetEdges() const
  {
    return edges;
  }

  // Get number of edges
  int GetNEdges() const
  {
    return edges.n_cols;
  }
  
  // Get vertex coordinates: size of #vertices * 3
  const arma::mat& GetVertexCoords() const
  {
    return vertexCoords;
  }

  // Set vertex coordinates. Need to re-compute dependent attributes
  virtual void SetVertexCoords(const arma::mat& vtCoords);

  // Get number of vertices
  int GetNVertices() const
  {
    return this->vertexCoords.n_rows;
  }

  // Get number of vertices
  int GetNFacets() const
  {
    return this->facets.n_rows;
  }

  // Get all pairs of adjacent facets
  const arma::umat& GetFacetPairs() const
  {
    return facetPairs;
  }

  // Get number of facet pairs
  int GetNFacetPairs() const
  {
    return facetPairs.n_rows;
  }

  // Get the vectorized coordinates of the mesh vertices with the order 
  // [x1,...,xN,y1,...,yN,z1,...,zN]
  const arma::vec& GetXYZColumn() const
  {
    return XYZColumn;
  }

  // Get edge lengths
  const arma::vec& GetEdgeLengths() const
  {
    // Be sure that edge lengths were already computed
    assert(edgeLengths.n_elem != 0);
    return this->edgeLengths;
  }

  // Compute edge lengths. Update edge length and return the reference as well
  const arma::vec& ComputeEdgeLengths();

  const arma::mat& GetFacetNormals() const
  {
    return this->facetNormals;
  }

  // Compute facet normals
  void computeFacetNormalsNCentroids();

protected:

  // Compute edges from vertices and facets
  void computeEdges();

  // Compute neighborhood relations from facets
  // Return a boolean matrix: 1 mean true, 0 mean false
  arma::SpMat<char> computeVertexNeighborhoods();

  // Compute facet pairs that share an edge
  void computeFacetPairs();
};

