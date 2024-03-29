//=============================================================================

#include "cstdio"
#include "cstdlib"
#include "cstdint"
#include "cstring"
#include "sys/time.h"
#include "errno.h"

#include "mpi.h"

#ifdef USE_GPU
#include "hip/hip_runtime.h"
#include "hipblas.h"
#include "hip/hip_fp16.h"
#else
#define __host__
#define __device__
typedef int16_t __half;
typedef int hipStream_t;
static __half __float2half(float i) {return i;}
typedef int hipblasDatatype_t;
typedef int hipblasHandle_t;
#endif

//=============================================================================

#define ASSERT(condition) \
  (void)((condition) || (assert_(#condition, __FILE__, __LINE__), 0))

void assert_(const char* condition_string, const char* file, int line) {
  fprintf(stderr, "%s: \"%s\". At file %s, line %i.\n", "Assertion error",
          condition_string, file, line);
  exit(EXIT_FAILURE);
}

#define SAFE_CALL_MPI(call) \
 {int errcode = call; \
  ASSERT(MPI_SUCCESS == errcode && "Failure in call: " #call);}

#ifdef USE_GPU

#define SAFE_CALL_GPU(call) \
 {hipError_t errcode = call; \
  ASSERT(hipSuccess == errcode && "Failure in call: " #call);}

#define SAFE_CALL_GPUBLAS(call) \
 {hipblasStatus_t errcode = call; \
  ASSERT(HIPBLAS_STATUS_SUCCESS == errcode && "Failure in call: " #call);}

#else

#define SAFE_CALL_GPU(call) {}
#define SAFE_CALL_GPUBLAS(call) {}

#endif

//-----------------------------------------------------------------------------

/// Wallclock timer.

double get_time() {

  struct timeval tv;
  gettimeofday(&tv, NULL);
  double result = ((double)tv.tv_sec + (double)tv.tv_usec * 1.e-6);

  return result;
}

//-----------------------------------------------------------------------------

/// GPU compute capability.

int compute_capability() {
#ifdef USE_GPU
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, 0); // Assume only one GPU per rank.
  return deviceProp.major * 100 + deviceProp.minor;
#else
  return 0;
#endif
}

//-----------------------------------------------------------------------------

/// Choices for tensor core GEMM method.

enum {
  TC_METHOD_NONE = 0,
  TC_METHOD_FLOAT16 = 1,
  TC_METHOD_INT8 = 2,
  TC_METHOD_FLOAT32 = 3,
  NUM_TC_METHOD = 4
};

//-----------------------------------------------------------------------------

template<typename GemmIn_t> struct TCBufTypes;

template<> struct TCBufTypes<__half> {
  static __host__ __device__ __half zero() {return __float2half(0.);}
  static __host__ __device__ __half one() {return __float2half(1.);}
  static __host__ __device__ __half two() {return __float2half(2.);}
};

//----------

template<> struct TCBufTypes<int8_t> {
  static __host__ __device__ int8_t zero() {return (int8_t)0;}
  static __host__ __device__ int8_t one() {return (int8_t)1;}
  static __host__ __device__ int8_t two() {return (int8_t)2;}
};

//----------

template<> struct TCBufTypes<float> {
  static __host__ __device__ float zero() {return (float)0;}
  static __host__ __device__ float one() {return (float)1;}
  static __host__ __device__ float two() {return (float)2;}
};

//-----------------------------------------------------------------------------

template<int TC_METHOD> struct TCSelector;

template<> struct TCSelector<TC_METHOD_INT8> {
  enum {TC_METHOD = TC_METHOD_INT8};
  // types.
  typedef int8_t GemmIn_t;
  typedef int32_t GemmOut_t;
#ifdef USE_GPU
  // type selector parameters.
  static hipblasDatatype_t __host__ __device__ gemm_type_in() {return HIPBLAS_R_8I;}
  static hipblasDatatype_t __host__ __device__ gemm_type_out() {return HIPBLAS_R_32I;}
#endif
};

//----------

template<> struct TCSelector<TC_METHOD_FLOAT16> {
  enum {TC_METHOD = TC_METHOD_FLOAT16};
  // types.
  typedef __half GemmIn_t;
  typedef float GemmOut_t;
#ifdef USE_GPU
  // type selector parameters.
  static hipblasDatatype_t __host__ __device__ gemm_type_in() {return HIPBLAS_R_16F;}
  static hipblasDatatype_t __host__ __device__ gemm_type_out() {return HIPBLAS_R_32F;}
#endif
};

//----------

template<> struct TCSelector<TC_METHOD_FLOAT32> {
  enum {TC_METHOD = TC_METHOD_FLOAT32};
  // types.
  typedef float GemmIn_t;
  typedef float GemmOut_t;
#ifdef USE_GPU
  // type selector parameters.
  static hipblasDatatype_t __host__ __device__ gemm_type_in() {return HIPBLAS_R_32F;}
  static hipblasDatatype_t __host__ __device__ gemm_type_out() {return HIPBLAS_R_32F;}
#endif
};

//-----------------------------------------------------------------------------

/// Matrix class, templated on scalar data type.

template<typename P_>
class Matrix {

  enum {ROUNDUP = 8};

  public:

    typedef P_ P;

    //----------

    Matrix(size_t num_row, size_t num_col)
      : num_row_(num_row)
      , num_col_(num_col)
      , num_row_up_(((num_row+ROUNDUP-1)/ROUNDUP)*ROUNDUP)
      , num_col_up_(((num_col+ROUNDUP-1)/ROUNDUP)*ROUNDUP)
      , num_elt_up_(num_row_up_ * num_col_up_)
      , sizeP(sizeof(P)) {

#ifdef USE_GPU
      SAFE_CALL_GPU(hipHostMalloc((void**)&h_, num_elt_up_ * sizeP));
      ASSERT(h_ && "Failure in host memory allocation");
      memset((void*)h_, 0, num_elt_up_ * sizeP);
      SAFE_CALL_GPU(hipMalloc((void**)&d_, num_elt_up_ * sizeP));
      ASSERT(d_ && "Failure in device memory allocation");
      SAFE_CALL_GPU(hipMemset((void*)h_, 0, num_elt_up_ * sizeP));
#else
      h_ = (P*)malloc(num_elt_up_ * sizeP);
      ASSERT(h_ && "Failure in host memory allocation");
      memset((void*)h_, 0, num_elt_up_ * sizeP);
#endif

    }

    //----------

    ~Matrix() {
#ifdef USE_GPU
      SAFE_CALL_GPU(hipHostFree(h_));
      SAFE_CALL_GPU(hipFree(d_));
#else
      free(h_);
#endif
    }

    //----------

    P* h() const {return h_;}
    P* d() const {return d_;}

    __host__ __device__ size_t nr() const {return num_row_;}
    __host__ __device__ size_t nc() const {return num_col_;}

    size_t nru() const {return num_row_up_;}
    size_t ncu() const {return num_col_up_;}

    //----------

    P& elt(size_t i, size_t j) {
      return h_[i + num_row_up_ * j];
    }

    //----------

    __device__ P& eltd(size_t i, size_t j) {
      return d_[i + num_row_up_ * j];
    }

    //----------

    void to_device(hipStream_t stream) {
      SAFE_CALL_GPU(hipMemcpyAsync(d_, h_, num_elt_up_ * sizeP,
        hipMemcpyHostToDevice, stream));
    }

    //----------

    void from_device(hipStream_t stream) {
      SAFE_CALL_GPU(hipMemcpyAsync(h_, d_, num_elt_up_ * sizeP,
        hipMemcpyDeviceToHost, stream));
    }

    //----------

    static __device__ P& eltd(size_t i, size_t j, P* d, size_t num_row_up) {
      return d[i + num_row_up * j];
    }

    //----------

  private:

    size_t num_row_;
    size_t num_col_;
    size_t num_row_up_;
    size_t num_col_up_;
    size_t num_elt_up_;
    size_t sizeP;

    P* h_;
    P* d_;

    // Disallowed methods.
    Matrix(const Matrix&);
    void operator=(const Matrix&);
};

//=============================================================================

/// Greatest common divisor.

size_t gcd(size_t a, size_t b){
  if (a == 0)
    return b;
  return gcd(b % a, a);
 }

//-----------------------------------------------------------------------------

/// Least common multiple.

size_t lcm(size_t a, size_t b){
  return (a * b) / gcd(a, b);
}

//-----------------------------------------------------------------------------

/// Distance between nonzero elements along a column of the matrix.

__host__ __device__ size_t nonzero_stride(const size_t& i) {
  enum {MAX = 499}; // Use prime number to randomize against sizes.
  return 1 + i % MAX;
}

//-----------------------------------------------------------------------------

/// HIP kernel for set_input_matrix.

#ifdef USE_GPU
template<class Matrix_t>
__global__ void set_input_matrix_kernel(Matrix_t& a,
  size_t nr, size_t nc, size_t nru, typename Matrix_t::P* d,
  size_t base_vector_num, typename Matrix_t::P value) {

  const size_t index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index >= nr * nc)
    return;

  const size_t r = index % nr;
  const size_t c = index / nr;

  typedef typename Matrix_t::P P;
  const P zero = TCBufTypes<P>::zero();

  const size_t stride = nonzero_stride(r + base_vector_num);

  Matrix_t::eltd(r, c, d, nru) = c % stride ? zero : value;
}
#endif

//-----------------------------------------------------------------------------

/// Set a sparse subset of the entries of a matrix.
///
/// All entries of the matrix A are zero, except for a small number of entries
/// along each column set to 1 according to a stride.  The number of
/// interactions of elements between two columns is based on the least common
/// multiple of their respective stride values.

template<class Matrix_t>
void set_input_matrix(Matrix_t& a, size_t base_vector_num,
       typename Matrix_t::P value, hipStream_t stream) {

#ifdef USE_GPU
  const int threadblocksize = 256;
  const int num_threadblocks = (a.nr() * a.nc() + threadblocksize - 1)
                               / threadblocksize;

  hipLaunchKernelGGL(HIP_KERNEL_NAME(set_input_matrix_kernel<Matrix_t>), dim3(num_threadblocks), dim3(threadblocksize), 0, stream , a, a.nr(), a.nc(), a.nru(), a.d(), base_vector_num, value);
#else
  for (size_t r=0; r<a.nr(); ++r) {
    const size_t stride = nonzero_stride(r + base_vector_num);
    for (size_t c=0; c<a.nc(); c+=stride) {
      a.elt(r, c) = value;
    }
  }
  //a.to_device(stream);
#endif
}

//-----------------------------------------------------------------------------

/// A very simplistic hash for a reult matrix element, used for validation.

size_t elt_hash(size_t v, size_t r, size_t c) {
  return 1 + (v * r * c) % (((size_t)1) << 40);
}

//-----------------------------------------------------------------------------

template<typename TCS, typename GemmIn_t, typename GemmOut_t>
void perform_gemm(hipblasHandle_t accelblas_handle, size_t m, size_t n, size_t k,
  Matrix<GemmIn_t>& tc_buf_left, Matrix<GemmIn_t>& tc_buf_right,
  Matrix<GemmOut_t>& c_buf) {

#ifdef USE_GPU
  // hipBLAS case.

  const GemmOut_t alpha = TCBufTypes<GemmOut_t>::one();
  const GemmOut_t beta = TCBufTypes<GemmOut_t>::zero();

  if (compute_capability() >= 700) {

    hipblasStatus_t status = hipblasGemmEx(
        accelblas_handle
      , HIPBLAS_OP_N, HIPBLAS_OP_T
      , m, n, k
      , (void*)&alpha
      , tc_buf_left.d(), TCS::gemm_type_in(), tc_buf_left.nru()
      , tc_buf_right.d(), TCS::gemm_type_in(), tc_buf_right.nru()
      , (void*)&beta
      , c_buf.d(), TCS::gemm_type_out(), c_buf.nru()
      , TCS::gemm_type_out()
      , HIPBLAS_GEMM_DEFAULT // changing from CUBLAS_GEMM_ALGO4_TENSOR_OP as hip has no equivalent
    );
    if (status == HIPBLAS_STATUS_NOT_INITIALIZED) {
      printf("Error: HIPBLAS_STATUS_NOT_INITIALIZED\n");
    } else if (status == HIPBLAS_STATUS_ARCH_MISMATCH) {
      printf("Error: HIPBLAS_STATUS_ARCH_MISMATCH\n");
    } else if (status == HIPBLAS_STATUS_NOT_SUPPORTED) {
      printf("Error: HIPBLAS_STATUS_NOT_SUPPORTED\n");
    } else if (status == HIPBLAS_STATUS_INVALID_VALUE) {
      printf("Error: HIPBLAS_STATUS_INVALID_VALUE\n");
    } else if (status == HIPBLAS_STATUS_EXECUTION_FAILED) {
      printf("Error: HIPBLAS_STATUS_EXECUTION_FAILED\n");
    }
    ASSERT(status == HIPBLAS_STATUS_SUCCESS && "Failure in call to hipblasGemmEx.");

  } else { // compute_capability

    hipblasStatus_t status = hipblasSgemm(
        accelblas_handle
      , HIPBLAS_OP_N, HIPBLAS_OP_T
      , m, n, k
      , &alpha
      , (float*)tc_buf_left.d(), tc_buf_left.nru()
      , (float*)tc_buf_right.d(), tc_buf_right.nru()
      , &beta
      , (float*)c_buf.d(), c_buf.nru()
    );
    ASSERT(status == HIPBLAS_STATUS_SUCCESS && "Failure in call to hipblasSgemm.");

  } // compute_capability

#else
  // Standard (C)BLAS case.

  ASSERT((int)TCS::TC_METHOD == (int)TC_METHOD_FLOAT32 && "Only SGEMM supported on CPU.");

  const float alpha = 1;
  const float beta = 0;

  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
    m, n, k, alpha, tc_buf_left.h(), tc_buf_left.nru(),
    tc_buf_right.h(), tc_buf_right.nru(), beta, c_buf.h(), c_buf.nru());
#endif
}

//-----------------------------------------------------------------------------

template<int TC_METHOD>
void perform_run(size_t num_vector, size_t num_field, int num_iterations) {

  SAFE_CALL_MPI(MPI_Barrier(MPI_COMM_WORLD));
  const double timetotal1 = get_time();

  int num_proc = 0;
  int proc_num = 0;
  SAFE_CALL_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &proc_num));
  SAFE_CALL_MPI(MPI_Comm_size(MPI_COMM_WORLD, &num_proc));

  // Compute sizes.

  // Because of divisibility issues, each proc may have a different number
  // of vectors.  However for simplicity the GEMM is computed on a padded-up
  // size that is the same on each proc.

  const size_t base_vector_num_left = (num_vector * proc_num) / num_proc;
  const size_t base_vector_num_leftp1 = (num_vector * (proc_num+1)) / num_proc;
  const size_t num_vector_local = base_vector_num_leftp1 - base_vector_num_left;
  const size_t num_vector_local_up = (num_vector + num_proc - 1) / num_proc;

  const size_t num_field_local = num_field;

  if (proc_num == 0) {
    printf("num_vector %zu num_field %zu num_iterations %i num_proc %i\n",
           num_vector, num_field, num_iterations, num_proc);
  }

  // CUDA initializations.

  hipStream_t stream;
  SAFE_CALL_GPU(hipStreamCreate(&stream));

  hipblasHandle_t accelblas_handle;
  SAFE_CALL_GPUBLAS(hipblasCreate(&accelblas_handle));
// Commenting out cublasSetMathMode as it is not supported in HIP.
// The library will decide that automatically
//  if (compute_capability() >= 700) {
//    SAFE_CALL_GPUBLAS(cublasSetMathMode(accelblas_handle,
//      CUBLAS_TENSOR_OP_MATH));
//  }
  SAFE_CALL_GPUBLAS(hipblasSetStream(accelblas_handle, stream));

  // Matrix setup.

  typedef TCSelector<TC_METHOD> TCS;
  typedef typename TCS::GemmIn_t GemmIn_t;
  typedef typename TCS::GemmOut_t GemmOut_t;

  const GemmOut_t zero = TCBufTypes<GemmOut_t>::zero();
  const GemmOut_t one = TCBufTypes<GemmOut_t>::one();

  const size_t m = 2 * num_vector_local_up; // each vec gets 2 matrix rows.
  const size_t n = m;
  const size_t k = num_field_local;

  Matrix<GemmIn_t> tc_buf_left(m, k);
  Matrix<GemmIn_t> tc_buf_right(n, k);
  Matrix<GemmOut_t> c_buf(m, n);

  set_input_matrix(tc_buf_left, base_vector_num_left, one, stream);
  c_buf.to_device(stream);

  // Loop over steps.

  double timegemm = 0;
  double flops_local = 0;
  size_t hash_local = 0;
  const int num_steps = (num_proc + 2) / 2;
  const int num_steps_this_proc = num_proc % 2 == 0 && proc_num >= num_proc/2 ?
    num_steps - 1 : num_steps;

  for (int iteration = 1; iteration <= num_iterations; ++iteration) {

    for (int step = 1; step <= num_steps; ++step) {

      SAFE_CALL_GPU(hipStreamSynchronize(stream));
      SAFE_CALL_MPI(MPI_Barrier(MPI_COMM_WORLD));
      const double timetotal2 = get_time();
      const double timetotal= timetotal2 - timetotal1;

      const bool do_out = proc_num == 0 && (
        !(iteration & (iteration-1)) || iteration % 256 == 0 ||
        iteration == num_iterations);

      if (do_out) {
        printf("Iteration %i of %i, step %i of %i, elapsed sec %.3f: setup...",
               iteration, num_iterations, step, num_steps, timetotal);
        fflush(stdout);
      }

      const int proc_num_right = (proc_num + step - 1) % num_proc;
      //const size_t base_vector_num_right = num_vector_local * proc_num_right;
      const size_t base_vector_num_right =
        (num_vector * proc_num_right) / num_proc;
      const size_t base_vector_num_rightp1 =
        (num_vector * (proc_num_right+1)) / num_proc;
      const size_t num_vector_local_right =
        base_vector_num_rightp1 - base_vector_num_right;

      const bool is_step_active = step <= num_steps_this_proc;

      if (is_step_active) {
        set_input_matrix(tc_buf_right, base_vector_num_right, one, stream);
      } // if is_step_active

      // Perform GEMM.

      if (do_out) {
        printf(" GEMM...");
        fflush(stdout);
      }

      SAFE_CALL_GPU(hipStreamSynchronize(stream));
      SAFE_CALL_MPI(MPI_Barrier(MPI_COMM_WORLD));
      const double timegemm1 = get_time();

      if (is_step_active) {
        perform_gemm<TCS, GemmIn_t, GemmOut_t>(accelblas_handle, m, n, k,
          tc_buf_left, tc_buf_right, c_buf);
        flops_local += 2. * m * n * k;
      } // if is_step_active

      SAFE_CALL_GPU(hipStreamSynchronize(stream));
      SAFE_CALL_MPI(MPI_Barrier(MPI_COMM_WORLD));
      const double timegemm2 = get_time();

      timegemm += timegemm2 - timegemm1;

      // Check.

      if (do_out) {
        printf(" check...");
        fflush(stdout);
      }

      if (is_step_active) {
        c_buf.from_device(stream);
        SAFE_CALL_GPU(hipStreamSynchronize(stream));

        const int check_freq1 = 89; // spot check, for speed.
        const int check_freq2 = 113;

        for (size_t c=0; c<m; c+=check_freq1) {
          const size_t stride2 = nonzero_stride(c + base_vector_num_right);
          for (size_t r=0; r<m; r+=check_freq2) {
            const size_t stride1 = nonzero_stride(r + base_vector_num_left);
            // WARNING: lcm can be slow, is not O(1) complexity.
            const size_t l = lcm(stride1, stride2);
            const size_t value = c_buf.elt(r,c);
            ASSERT(c_buf.elt(r,c) == 1 + (k-1)/l && "Error in compiuted result.");
          }
        }

#ifndef USE_GPU
        // Reset right matrix to zero to prepare for next pass.
        // GPU case doesn't need because overwritten by next transfer.
        set_input_matrix(tc_buf_right, base_vector_num_right, zero, stream);
#endif
      } // if is_step_active

      // Compute hash/checksum.

      if (is_step_active) {
        for (size_t c=0; c<num_vector_local_right; ++c) {
          const size_t c_global = c + base_vector_num_right;
          for (size_t r=0; r<num_vector_local; ++r) {
            const size_t r_global = r + base_vector_num_left;
            const bool not_in_upper = step==1 && r >= c;
            if (not_in_upper)
              continue;
            const size_t value = c_buf.elt(r,c);
            hash_local += elt_hash(value, r_global, c_global);
            //printf("%zu %zu %zu\n", r_global, c_global, value);
          }
        }
      } // if is_step_active

      if (do_out) {
        printf("\n");
        fflush(stdout);
      }

    } // step

  } // for iteration

  // Print final reaults.

  double flops = 0;
  SAFE_CALL_MPI(MPI_Allreduce(&flops_local, &flops, 1,
                           MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));

  size_t hash = 0;
  SAFE_CALL_MPI(MPI_Allreduce(&hash_local, &hash, 1,
                           MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD));

  SAFE_CALL_GPU(hipStreamSynchronize(stream));
  SAFE_CALL_MPI(MPI_Barrier(MPI_COMM_WORLD));
  const double timetotal2 = get_time();
  const double timetotal= timetotal2 - timetotal1;

  if (proc_num == 0) {
    printf("TF %.3f GEMM sec %.3f GEMM TF/sec %.3f total sec %.3f hash %zu\n",
      flops/1e12, timegemm, flops*1e-12/timegemm, timetotal, hash);
  }

  // Finish.

  SAFE_CALL_GPUBLAS(hipblasDestroy(accelblas_handle));
  SAFE_CALL_GPU(hipStreamDestroy(stream));
}

//-----------------------------------------------------------------------------

int main(int argc, char** argv) {

  // Initialize MPI.

  SAFE_CALL_MPI(MPI_Init(&argc, &argv));

  // Parse command line.

  size_t num_vector = 0;
  size_t num_field = 0;
  int num_iterations = 1;

  for (int i = 1 ; i < argc; ++i) {
    if (strcmp(argv[i], "--num_vector") == 0) {
      ++i;
      ASSERT(i < argc && 0 ? 0 : "Missing value for num_vector.");
      num_vector = strtol(argv[i], NULL, 10);
    }
    if (strcmp(argv[i], "--num_field") == 0) {
      ++i;
      ASSERT(i < argc && 0 ? 0 : "Missing value for num_field.");
      num_field = strtol(argv[i], NULL, 10);
    }
    if (strcmp(argv[i], "--num_iterations") == 0) {
      ++i;
      ASSERT(i < argc && 0 ? 0 : "Missing value for num_iterations.");
      num_iterations = atoi(argv[i]);
    }
  } // i

  ASSERT(num_vector >= 2);
  ASSERT(num_field >= 1);
  ASSERT(num_iterations >= 1);

  if (compute_capability() >= 700) {
#ifdef USE_GPU
    perform_run<TC_METHOD_FLOAT16>(num_vector, num_field, num_iterations);
#endif
  } else {
    perform_run<TC_METHOD_FLOAT32>(num_vector, num_field, num_iterations);
  }

  // Finish.

  SAFE_CALL_MPI(MPI_Finalize());
  return 0;
}

//=============================================================================
