#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include <smmintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

//#define BASE
//#define V1 
//#define V2
#define V3
 
#define N 1
const char *usage = "%s <square matrix one dimension size>";

double rtclock()
{
  struct timezone Tzp;
  struct timeval Tp; 
  int stat;
  double t;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  return t = (Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

void Transpose(int size, float* m)
{
  int i, j;
  for (i = 0; i < size; i++) {
    for (j = i + 1; j < size; j++) {
      float temp;
      temp = m[i*size+j];
      m[i*size+j] = m[j*size+i];
      m[j*size+i] = temp;
    }
  }
}

#ifdef BASE

/*Baseline Matrix Multiplication*/
void mm(int size, float* A, float* B, float* C)
{
  int i, j, k;
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      C[i*size+j] = 0;
      for (k = 0; k < size; k++) {
        C[i*size+j] += A[i*size+k] * B[k*size+j];
      }
    }
  }
}

#elif defined V1

/*Opt1: transpose B to improve data locality*/
void mm(int size, float* A, float* B, float* C)
{
  int i, j, k;
  Transpose(size, B);
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      float c = 0;
      for (k = 0; k < size; k++) {
        c += A[i*size+k] * B[j*size+k];
      }
      C[i*size+j] = c;
    }
  }
  Transpose(size, B);
}

#elif defined V2

/*Opt2: transpose B and tile to improve data locality*/
void mm(int size, float* A, float* B, float* C)
{
	int tile_size = 1024; // test with 2, 4, 8, 16, 32, 64, 128, 256, 512, and 1024
	int ii, jj;
	  int i, j, k;
	  Transpose(size, B);
	  for (ii = 0; ii < size; ii += tile_size) {
	  	for (jj = 0; jj < size; jj += tile_size) {
	  		for (i = ii; i < ii + tile_size; i++) {
	  			for (j = jj; j < jj + tile_size; j++) {
	  				float c = 0;
				      for (k = 0; k < size; k++) {
				        c += A[i*size+k] * B[j*size+k];
				      }
				      C[i*size+j] = c;
	  			}
	  		}
	  	}
	  }
	  Transpose(size, B);
}

#elif defined V3

/*Opt3: transpose B and vectorize*/
void mm(int size, float* A, float* B, float* C)
{
	__m128 A_vec, B_vec, C_vec; // initialize 4 single precision floating point vectors
	__m128 res_vec; // the result vector
	__m128 zero_vec = _mm_setzero_ps(); // Create vector of 4 zeros
	__m128 tmp_vec;

	int i, j, k;
    Transpose(size, B);
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            res_vec = _mm_setzero_ps(); // set res_vec to vector of 4 zeros
            for (k = 0; k < size; k+=4) {
            	A_vec = _mm_load_ps(&A[i*size+k]);
            	B_vec = _mm_load_ps(&B[j*size+k]);
            	C_vec = _mm_mul_ps(A_vec, B_vec);
            	tmp_vec = _mm_add_ps(res_vec, C_vec);
                res_vec = tmp_vec;
            }
            tmp_vec = _mm_hadd_ps(_mm_hadd_ps(res_vec, zero_vec), zero_vec);
            _mm_store_ss(&C[i*size+j], tmp_vec);
        }
    }
    Transpose(size, B);
}

#endif

int main(int argc, char *argv[]) {
  int msize;
  __attribute__((aligned(16)))  float *A, *B, *C;
  double begin, last;
  double cputime, gflops;
  int i;

  if (argc!=2) {
    printf(usage, argv[0]);
    return 0;
  }

  msize = atoi(argv[1]);

  assert(msize>0);
  
  A = (float *) _mm_malloc (msize * msize * sizeof(float), 16);
  B = (float *) _mm_malloc (msize * msize * sizeof(float), 16);
  C = (float *) _mm_malloc (msize * msize * sizeof(float), 16);

  assert(A && B && C);

  memset(A, 0, msize * msize * sizeof(float));
  memset(B, 0, msize * msize * sizeof(float));
  memset(C, 0, msize * msize * sizeof(float));

  int ii,jj;
  for(ii = 0; ii < msize; ii++){
    for (jj = 0; jj < msize; jj++){
      A[ii*msize + jj] = (ii + jj) * 0.1;
    }
  }

  for(ii = 0; ii < msize; ii++){
    for (jj = 0; jj < msize; jj++){
      B[ii*msize + jj] = (ii - jj) * 0.2;
    }
  }
  
  float rets = 0.0;
  begin = rtclock();
  
  for(i=0; i<N; i++) {
    mm(msize, A, B, C);
    rets += C[i];
  }

  last = rtclock();
  cputime = (last - begin) / N;

  gflops = (2.0 * msize * msize * msize / cputime) / 1000000000.0;
  printf("Time=%lfms GFLOPS=%.3lf\n", cputime*1000, gflops);
  printf("Prove optimization out: rets = %lf\n", rets);

  _mm_free(A);
  _mm_free(B);
  _mm_free(C);
  return 0;  
}
