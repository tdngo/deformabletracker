//////////////////////////////////////////////////////////////////////////
// Author       :  Ngo Tien Dat
// Email        :  dat.ngo@epfl.ch
// Organization :  EPFL
// Purpose      :  Laplacian mesh class that inherits from TrigMesh
// Date         :  13 March 2012
//////////////////////////////////////////////////////////////////////////

#pragma once

#include "TriangleMesh.h"
#include <Linear/LinearAlgebraUtils.h>

// LaplacianMesh inherits from TrigMesh
class LaplacianMesh : public TriangleMesh
{

protected:
  // Attributes

  arma::urowvec ctrPointIds;   // List of control point ids
  arma::mat     ctrlVertices;  // List of control point coordinates.
                               // Is not used in any methods (except GetCtrlVertices()) of this class.

  arma::mat    regMat;         // Regularization matrix
  arma::mat    paramMat;       // Parameterization matrix, size of #vertices x #control points
  
  // Assemble A, P to make bigger regularization and parameterization matrix
  //  A 0 0    P 0 0
  //  0 A 0    0 P 0
  //  0 0 A    0 0 P
  arma::mat    bigReglMat;     // Three regMats are put in the diagonal of a 3 times bigger matrix
  arma::mat    bigParamMat;    // Three paramMats are put in the diagonal of a 3 times bigger matrix
  arma::mat    bigAP;          // Product bigA * bigP, precomputed to avoid re-computing
  
public:

  // Empty constructor
  LaplacianMesh() {}

  // Constructor that creates a new Laplacian mesh containing only 
  // controlPointIds and edges information. Avoid copying other matrices
  // Used to store reconstruction result
  LaplacianMesh(const arma::urowvec& pCtrPointIds, const arma::umat& pEdges)
  {
    this->ctrPointIds  = pCtrPointIds;
    this->edges      = pEdges;
  }

  // Compute all A, P matrices
  void ComputeAPMatrices();

  void LoadAPMatrices(std::string dataFolder);

  // Set vertex coordinates. Need to (update) reset dependent attributes
  virtual void SetVertexCoords(const arma::mat& vtCoords);

  // Set control points for the Laplacian mesh. Ids of control vertices start from 0
  void SetCtrlPointIDs(arma::urowvec pCtrPointIDs);

  // Get control point ids
  const arma::urowvec& GetCtrlPointIDs() const
  {
    return this->ctrPointIds;
  }

  // Get regularization matrix A
  const arma::mat& GetRegMat() const
  {
    return regMat;
  }

  const arma::mat& GetBigReglMat() const
  {
    return bigReglMat;
  }
  
  // Get parameterization matrix
  const arma::mat& GetParamMatrix() const
  {
    return paramMat;
  }

  const arma::mat& GetBigParamMat() const
  {
    return bigParamMat;
  }

  // Get coordinates of control points
  // Size of #control_points * 3
  const arma::mat& GetCtrlVertices()
  {
    // If already computed
    if (ctrlVertices.n_elem > 0)
      return ctrlVertices;

    // Else, compute and return control vertex coordinates
    int nCtrlPoints = this->GetNCtrlPoints();
    this->ctrlVertices.set_size(nCtrlPoints, 3);    
    for (int i = 0; i < nCtrlPoints; i++)
    {
      ctrlVertices.row(i) = this->vertexCoords.row(ctrPointIds(i));
    }
    
    return this->ctrlVertices;
  }
  
  // Get matrix bigAP
  const arma::mat& GetBigAP() const
  {
    return this->bigAP;
  }

  // Return the number of control points
  int GetNCtrlPoints() const
  {
    return this->ctrPointIds.n_elem;
  }

private:
  // Virtual function to be overridden by the child classes
  // Compute regularization matrix
  virtual void computeRegMatrix();

  // Compute parameterization matrix P
  void computeParamMatrix();

  // An utility function that computes the ordered union of vertex ids of two adjacent vertices.
  // Input:
  //  + facet1: 1x3 
  //  + facet2: 1x3 
  // Let a b c are the vertex Ids of facet 1, d c b are vertex Ids of facet 2
  // 
  // Output:
  //  + The function should return the result in order: a b c d, meaning, two shared 
  //  vertices are in the middle
  static arma::urowvec getOrderedVertexIds(arma::urowvec facet1, arma::urowvec facet2);

  // Compute Y from X. Need documentation???
  // X represents coordinates of four vertices of two adjacent facets
  // X has 4 rows, 3 columns. 
  static arma::mat unfoldFacetPair(arma::mat X);
  
  // ltc function??? Documentation
  static arma::rowvec ltc(double a, double b, double c, int sign);

  // Compute weight w1, w2, w3, w4 to put into regularization matrix
  static arma::vec computeWeights(arma::mat Y);

  // Compute Pc matrix so that Pc * X = C
  arma::mat computePcMatrix() const;
};

