#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include <smmintrin.h>

#define BASE
//#define V1 
//#define V2
//#define V3
 
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

}

#elif defined V3

/*Opt3: transpose B and vectorize*/
void mm(int size, float* A, float* B, float* C)
{

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
