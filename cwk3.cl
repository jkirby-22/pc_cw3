//kernel if matrix exceeds devices constant capacity
__kernel
void transposeGlobal( __global float *oldMatrix, __global float *newMatrix, int nRows, int nCols) //oldMatrix in global, local copy would be faster but we want to keep kernel small
{
	// Store in private variables to optimise multiple access
	int rowIndex = get_global_id(0);
  int columnIndex = get_global_id(1);

  //transpose
	//newMatrix[rowIndex] = rowIndex;
  newMatrix[(columnIndex * nRows) + rowIndex] = oldMatrix[(rowIndex * nCols) + columnIndex];

}

//optimal kernel when matrix within constant capacity
__kernel
void transposeConstant( __constant float *oldMatrix, __global float *newMatrix, int nRows, int nCols) //oldMatrix in constant, local copy would be faster but we want to keep kernel small
{
	// Store in private variables to optimise multiple access
	int rowIndex = get_global_id(0);
  int columnIndex = get_global_id(1);

  //transpose
	//newMatrix[rowIndex] = rowIndex;
  newMatrix[(columnIndex * nRows) + rowIndex] = oldMatrix[(rowIndex * nCols) + columnIndex];

}
