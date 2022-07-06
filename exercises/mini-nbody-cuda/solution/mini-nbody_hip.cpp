#include "hip/hip_runtime.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Macro for checking errors in CUDA API calls
#define hipErrorCheck(call)                                                              \
do{                                                                                       \
    hipError_t cuErr = call;                                                             \
    if(hipSuccess != cuErr){                                                             \
      printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(cuErr));\
      exit(0);                                                                            \
    }                                                                                     \
}while(0)

#define BLOCK_SIZE 256
#define SOFTENING 1e-9f

typedef struct { float x, y, z, vx, vy, vz; } Body;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

__global__
void bodyForce(Body *p, float dt, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}

int main(const int argc, const char** argv) {
  
  int nBodies = 100000;
  if (argc > 1) nBodies = atoi(argv[1]);
  
  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations

  int bytes = nBodies*sizeof(Body);
  float *buf = (float*)malloc(bytes);
  Body *p = (Body*)buf;

  randomizeBodies(buf, 6*nBodies); // Init pos / vel data

  float *d_buf;
  hipErrorCheck( hipMalloc(&d_buf, bytes) );
  Body *d_p = (Body*)d_buf;

  int nBlocks = (nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;

  double total_time_ms = 0.0;

  hipEvent_t start, stop;
  float time_ms;
  hipErrorCheck( hipEventCreate(&start) );
  hipErrorCheck( hipEventCreate(&stop) );


  for (int iter = 1; iter <= nIters; iter++) {

    hipErrorCheck( hipEventRecord(start, NULL) );

    hipErrorCheck( hipMemcpy(d_buf, buf, bytes, hipMemcpyHostToDevice) );
    hipLaunchKernelGGL(bodyForce, nBlocks, BLOCK_SIZE, 0, 0, d_p, dt, nBodies); // compute interbody forces
    hipErrorCheck( hipMemcpy(buf, d_buf, bytes, hipMemcpyDeviceToHost) );

    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }

    hipErrorCheck( hipEventRecord(stop, NULL) );
    hipErrorCheck( hipEventSynchronize(stop) );
    hipErrorCheck( hipEventElapsedTime(&time_ms, start, stop) );

    if (iter > 1) { // First iter is warm up
      total_time_ms += time_ms; 
    }

  printf("Iteration %d: %.3f seconds\n", iter, time_ms * 1e-3);

  }

  double avg_time_s = total_time_ms * 1e-3 / (double)(nIters-1);
  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avg_time_s);

  free(buf);
  hipErrorCheck( hipFree(d_buf) );
}
