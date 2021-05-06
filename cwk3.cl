// Kernel for matrix transposition.
__kernel
void matrixTranspose( __constant float *oldMatrix, __global float *newMatrix, int nRows, int nCols) //oldMatrix in constant, local copy would be faster but we want to keep kernel small
{
	// Store in private variables to optimise multiple access
	int rowIndex = get_global_id(0);
  int columnIndex = get_global_id(1);

  //transpose
  newMatrix[(columnIndex * nRows) + rowIndex] = oldMatrix[(rowIndex * nCols) + columnIndex];

}
