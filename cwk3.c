//
// Starting point for the OpenCL coursework for COMP3221 Parallel Computation.
//
// Once compiled, execute with the number of rows and columns for the matrix, e.g.
//
// ./cwk3 16 8
//
// This will display the matrix, followed by another matrix that has not been transposed
// correctly. You need to implement OpenCL code so that the transpose is correct.
//
// For this exercise, both the number of rows and columns must be a power of 2,
// i.e. one of 1, 2, 4, 8, 16, 32, ...
//


//
// Includes.
//
#include <stdio.h>
#include <stdlib.h>

// For this coursework, the helper file has 3 routines in addition to simpleOpenContext_GPU() and compileKernelFromFile():
// - getCmdLineArgs(): Gets the command line arguments and checks they are valid.
// - displayMatrix() : Displays the matrix, or just the top-left corner if it is too large.
// - fillMatrix()    : Fills the matrix with random values.
// Do not alter these routines, as they will be replaced with different versions for assessment.
#include "helper_cwk.h"


//
// Main.
//
int main( int argc, char **argv )
{
    //
    // Parse command line arguments and check they are valid. Handled by a routine in the helper file.
    //
    int nRows, nCols;
    getCmdLineArgs( argc, argv, &nRows, &nCols );

    //
    // Initialisation.
    //

    // Set up OpenCL using the routines provided in helper_cwk.h.
    cl_device_id device;
    cl_context context = simpleOpenContext_GPU(&device);

    // Open up a single command queue, with the profiling option off (third argument = 0).
    cl_int status;
    cl_command_queue queue = clCreateCommandQueue( context, device, 0, &status );

    // Allocate memory for the matrix.
    float *hostMatrix = (float*) malloc( nRows*nCols*sizeof(float) );


    // Fill the matrix with random values, and display.
    fillMatrix( hostMatrix, nRows, nCols );
    printf( "Original matrix (only top-left shown if too large):\n" );
    displayMatrix( hostMatrix, nRows, nCols );

    //create buffers
    cl_mem device_oldMatrix = clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nRows*nCols*sizeof(float), hostMatrix, &status );
  	cl_mem device_newMatrix = clCreateBuffer( context, CL_MEM_WRITE_ONLY, nRows*nCols*sizeof(float), NULL, &status );

    cl_kernel kernel = compileKernelFromFile( "cwk3.cl", "matrixTranspose", context, device );

    //set kernel arguments
    status = clSetKernelArg( kernel, 0, sizeof(cl_mem), &device_oldMatrix );
    status = clSetKernelArg( kernel, 1, sizeof(cl_mem), &device_newMatrix );
    status = clSetKernelArg( kernel, 2, sizeof(int), &nRows );
    status = clSetKernelArg( kernel, 3, sizeof(int), &nCols );



    size_t indexSpaceSize[2] = {nRows, nCols};
    //let automatic handling of work group size to ensure compatibility with index space size
  	status = clEnqueueNDRangeKernel( queue, kernel, 2, 0, indexSpaceSize, NULL, 0, NULL, NULL );
    status = clEnqueueReadBuffer( queue, device_newMatrix, CL_TRUE, 0, nRows*nCols*sizeof(float), hostMatrix, 0, NULL, NULL );

    printf( "Transposed matrix (only top-left shown if too large):\n" );
    displayMatrix( hostMatrix, nCols, nRows );


    //
    // Release all resources.
    //
    clReleaseCommandQueue( queue   );
    clReleaseContext     ( context );
    clReleaseKernel      ( kernel  );

    clReleaseMemObject( device_oldMatrix );
    clReleaseMemObject( device_newMatrix );

    free( hostMatrix );

    return EXIT_SUCCESS;
}
