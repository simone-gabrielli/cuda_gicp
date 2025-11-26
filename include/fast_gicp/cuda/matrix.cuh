#ifndef CUGICP_MATH_CUH
#define CUGICP_MATH_CUH

#include <cuda_runtime.h>
#include <cmath>

/**
 * @brief Performs multiplication of two NxN matrices.
 *
 * Computes C = A * B for matrices stored in row–major order.
 *
 * @tparam N The dimension of the matrices.
 * @param A Pointer to the first matrix (size N*N).
 * @param B Pointer to the second matrix (size N*N).
 * @param C Pointer to the output matrix (size N*N).
 */
template <int N>
__host__ __device__ inline void matMul(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C)
{
    float temp[N * N];
    #pragma unroll
    for (int row = 0; row < N; ++row) {
        #pragma unroll
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            temp[row * N + col] = sum;
        }
    }
    #pragma unroll
    for (int i = 0; i < N * N; ++i) {
        C[i] = temp[i];
    }
}

/**
 * @brief Inverts an NxN matrix using Gaussian elimination with partial pivoting.
 *
 * The matrix is provided in row–major order. Returns false if the matrix is singular.
 *
 * @tparam N The dimension of the matrix.
 * @param A Pointer to the input matrix (size N*N).
 * @param A_inv Pointer to the output inverse matrix (size N*N).
 * @return true if inversion was successful; false if the matrix is singular.
 */
template <int N>
__host__ __device__ inline bool invertMatrix(const float* __restrict__ A,
                                              float* __restrict__ A_inv)
{
    float aug[N][2 * N];
    // Build augmented matrix [A | I]
    for (int i = 0; i < N; ++i) {
        #pragma unroll
        for (int j = 0; j < N; ++j) {
            aug[i][j] = A[i * N + j];
            aug[i][j + N] = (i == j) ? 1.0f : 0.0f;
        }
    }
    // Gaussian elimination with partial pivoting
    for (int i = 0; i < N; ++i) {
        int pivot = i;
        for (int j = i + 1; j < N; ++j) {
            if (fabsf(aug[j][i]) > fabsf(aug[pivot][i])) {
                pivot = j;
            }
        }
        if (fabsf(aug[pivot][i]) < 1e-6f) return false; // Singular matrix
        if (pivot != i) {
            for (int j = 0; j < 2 * N; ++j) {
                float tmp = aug[i][j];
                aug[i][j] = aug[pivot][j];
                aug[pivot][j] = tmp;
            }
        }
        float pivot_val = aug[i][i];
        for (int j = 0; j < 2 * N; ++j) {
            aug[i][j] /= pivot_val;
        }
        // Eliminate column i in other rows.
        for (int k = 0; k < N; ++k) {
            if (k != i) {
                float factor = aug[k][i];
                for (int j = 0; j < 2 * N; ++j) {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }
    // Copy inverse part to output.
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A_inv[i * N + j] = aug[i][j + N];
        }
    }
    return true;
}

/**
 * @brief Inverts an affine transformation matrix of size (N+1)x(N+1).
 *
 * The transformation matrix T is assumed to be of the form:
 * [ R | t ]
 * [ 0 | 1 ]
 * where R is an NxN rotation matrix (orthonormal) and t is an N×1 translation.
 * The inverse is computed as:
 * T_inv = [ R^T | -R^T * t ]
 *         [  0  |    1     ]
 *
 * @tparam N Dimension of the rotation part.
 * @param T Pointer to the input transformation matrix (size (N+1)*(N+1)) in row–major order.
 * @param T_inv Pointer to the output inverse transformation matrix (size (N+1)*(N+1)).
 */
template <int N>
__host__ __device__ inline void invertTransform(const float* __restrict__ T,
                                                  float* __restrict__ T_inv)
{
    float R_T[N * N];
    float t[N];
    // Extract translation (assumes t is in the last column of the first N rows)
    for (int i = 0; i < N; ++i) {
        t[i] = T[i * (N + 1) + N];
    }
    // Transpose rotation matrix.
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            R_T[i * N + j] = T[j * (N + 1) + i];
        }
    }
    // Compute -R^T * t.
    float t_inv[N];
    for (int i = 0; i < N; ++i) {
        t_inv[i] = 0.0f;
        for (int j = 0; j < N; ++j) {
            t_inv[i] -= R_T[i * N + j] * t[j];
        }
    }
    // Construct inverse transformation.
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            T_inv[i * (N + 1) + j] = R_T[i * N + j];
        }
        T_inv[i * (N + 1) + N] = t_inv[i];
    }
    // Set last row to [0 ... 0 1].
    for (int j = 0; j < N; ++j) {
        T_inv[N * (N + 1) + j] = 0.0f;
    }
    T_inv[N * (N + 1) + N] = 1.0f;
}

/**
 * @brief Constructs a 3x3 skew–symmetric matrix from a 3D vector.
 *
 * The resulting matrix satisfies skew(v) * w = v × w.
 *
 * @param v Input 3D vector.
 * @param skew Pointer to the output 3x3 matrix (row–major order).
 */
__host__ __device__ inline void skewSymmetric(const float3 &v,
                                               float* __restrict__ skew)
{
    skew[0] =  0.0f;  skew[1] = -v.z;  skew[2] =  v.y;
    skew[3] =  v.z;   skew[4] =  0.0f;  skew[5] = -v.x;
    skew[6] = -v.y;   skew[7] =  v.x;  skew[8] =  0.0f;
}

/**
 * @brief Computes the SO(3) exponential map from an angle–axis vector.
 *
 * Converts a 3D rotation represented as an angle–axis vector (omega) to a 3x3 rotation matrix.
 *
 * @param omega Input rotation vector (angle–axis, 3 elements).
 * @param rotation Pointer to the output 3x3 rotation matrix (row–major order).
 */
__host__ __device__ inline void so3_exp(const float omega[3],
                                        float* __restrict__ rotation)
{
    float theta_sq = omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2];
    float theta = sqrtf(theta_sq);

    if (theta_sq < 1e-10f) {
        rotation[0] = 1.0f; rotation[1] = 0.0f; rotation[2] = 0.0f;
        rotation[3] = 0.0f; rotation[4] = 1.0f; rotation[5] = 0.0f;
        rotation[6] = 0.0f; rotation[7] = 0.0f; rotation[8] = 1.0f;
        return;
    }

    float ux = omega[0] / theta;
    float uy = omega[1] / theta;
    float uz = omega[2] / theta;

    float c = cosf(theta);
    float s = sinf(theta);
    float one_c = 1.0f - c;

    rotation[0] = c + ux*ux*one_c;
    rotation[1] = ux*uy*one_c - uz*s;
    rotation[2] = ux*uz*one_c + uy*s;

    rotation[3] = uy*ux*one_c + uz*s;
    rotation[4] = c + uy*uy*one_c;
    rotation[5] = uy*uz*one_c - ux*s;

    rotation[6] = uz*ux*one_c - uy*s;
    rotation[7] = uz*uy*one_c + ux*s;
    rotation[8] = c + uz*uz*one_c;
}

/**
 * @brief Solves a linear system using LDL^T decomposition.
 *
 * Decomposes a symmetric matrix A (of dimension N×N) in place and solves (A + λI)d = -b.
 *
 * @tparam N The dimension of the matrix.
 * @param A Input matrix (will be modified in-place) of size N×N.
 * @param b Right–hand side vector of length N.
 * @param d Output solution vector of length N.
 */
template <int N>
__host__ __device__ inline void solveLDLT(float A[N][N], float b[N], float d[N])
{
    // LDL^T decomposition in place.
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < i; k++) {
            A[i][i] -= A[i][k] * A[i][k] * A[k][k];
        }
        for (int j = i + 1; j < N; j++) {
            for (int k = 0; k < i; k++) {
                A[j][i] -= A[j][k] * A[i][k] * A[k][k];
            }
            A[j][i] /= A[i][i];
        }
    }

    float y[N];
    for (int i = 0; i < N; i++) {
        y[i] = -b[i];
        for (int j = 0; j < i; j++) {
            y[i] -= A[i][j] * y[j];
        }
    }

    float z[N];
    for (int i = 0; i < N; i++) {
        z[i] = y[i] / A[i][i];
    }

    for (int i = N - 1; i >= 0; i--) {
        d[i] = z[i];
        for (int j = i + 1; j < N; j++) {
            d[i] -= A[j][i] * d[j];
        }
    }
}


/**
 * @brief Adds a scaled identity matrix to an NxN matrix.
 *
 * Computes B = A + lambda * I.
 *
 * @tparam N The dimension of the matrix.
 * @param A Input matrix.
 * @param lambda Scalar value to add on the diagonal.
 * @return The resulting matrix.
 */
template <int N>
__host__ __device__ inline void matAddIdentity(const float* __restrict__ A,
                                               float lambda,
                                               float* __restrict__ B)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B[i * N + j] = A[i * N + j];
        }
        B[i * N + i] += lambda;
    }
}

/**
 * @brief Computes the determinant of an NxN matrix.
 *
 * Uses Gaussian elimination with partial pivoting.
 *
 * @tparam N The dimension of the matrix.
 * @param A Input matrix.
 * @return The determinant of A.
 */
template <int N>
__host__ __device__ inline float matDet(const float* __restrict__ A)
{
    float M[N][N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            M[i][j] = A[i * N + j];

    float det = 1.0f;
    for (int i = 0; i < N; i++) {
        int pivot = i;
        for (int j = i + 1; j < N; j++) {
            if (fabsf(M[j][i]) > fabsf(M[pivot][i]))
                pivot = j;
        }
        if (fabsf(M[pivot][i]) < 1e-6f)
            return 0.0f;
        if (pivot != i) {
            for (int j = 0; j < N; j++) {
                float tmp = M[i][j];
                M[i][j] = M[pivot][j];
                M[pivot][j] = tmp;
            }
            det = -det;
        }
        det *= M[i][i];
        for (int j = i + 1; j < N; j++) {
            float factor = M[j][i] / M[i][i];
            for (int k = i; k < N; k++) {
                M[j][k] -= factor * M[i][k];
            }
        }
    }
    return det;
}

/**
 * @brief Computes the Frobenius norm of an NxN matrix.
 *
 * @tparam N The dimension of the matrix.
 * @param A Input matrix.
 * @return The Frobenius norm.
 */
template <int N>
__host__ __device__ inline float matFrobeniusNorm(const float* __restrict__ A)
{
    float sum = 0.0f;
    for (int i = 0; i < N * N; i++)
        sum += A[i] * A[i];
    return sqrtf(sum);
}

/**
 * @brief Transposes an NxN matrix.
 *
 * @tparam N The dimension of the matrix.
 * @param A Input matrix.
 * @return The transposed matrix.
 */
template <int N>
__host__ __device__ inline void matTranspose(const float* __restrict__ A,
                                             float* __restrict__ T)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            T[i * N + j] = A[j * N + i];
}

////////////////////////////////////////////////////////////////////////////////
// Note: The following eigen-decomposition routine is only implemented for 
// 3x3 symmetric matrices. A general NxN eigen-solver is not provided here.
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Performs eigen–decomposition of a symmetric 3x3 matrix using Jacobi iterations.
 *
 * Computes eigenvalues and eigenvectors of A. The eigenvectors are stored as columns in Q.
 *
 * @param A Input symmetric 3x3 matrix.
 * @param Q Output matrix whose columns are the eigenvectors.
 * @param eig Output array of eigenvalues (length 3).
 */
__host__ __device__ inline void eigen_decomposition(const float A[9], float Q[9], float eig[3])
{
    // Copy A into a mutable array.
    float a[9];
    for (int i = 0; i < 9; i++)
        a[i] = A[i];

    // Enforce symmetry to avoid numerical asymmetry issues.
    a[0] = 0.5f * (a[0] + a[0]);
    a[1] = 0.5f * (a[1] + a[3]);
    a[2] = 0.5f * (a[2] + a[6]);
    a[3] = a[1];
    a[4] = 0.5f * (a[4] + a[4]);
    a[5] = 0.5f * (a[5] + a[7]);
    a[6] = a[2];
    a[7] = a[5];
    a[8] = 0.5f * (a[8] + a[8]);

    // Initialize Q as the identity matrix.
    Q[0] = 1.0f; Q[1] = 0.0f; Q[2] = 0.0f;
    Q[3] = 0.0f; Q[4] = 1.0f; Q[5] = 0.0f;
    Q[6] = 0.0f; Q[7] = 0.0f; Q[8] = 1.0f;

    // Use more Jacobi iterations and better angle formula.
    const int maxIter = 50;
    const float tol = 1e-8f;
    for (int iter = 0; iter < maxIter; iter++) {
        // Find the largest off–diagonal element.
        int p = 0, q = 1;
        float maxOff = fabsf(a[0*3+1]);
        if (fabsf(a[0*3+2]) > maxOff) { maxOff = fabsf(a[0*3+2]); p = 0; q = 2; }
        if (fabsf(a[1*3+2]) > maxOff) { maxOff = fabsf(a[1*3+2]); p = 1; q = 2; }

        // If off–diagonals are small, consider the matrix diagonalized.
        if (maxOff < tol)
            break;

        // Compute the Jacobi rotation angle using atan2 for stability.
        float app = a[p*3+p];
        float aqq = a[q*3+q];
        float apq = a[p*3+q];
        float phi = 0.5f * atan2f(2.0f * apq, (aqq - app));
        float c = cosf(phi);
        float s = sinf(phi);

        // Update diagonal elements.
        float app_new = c * c * app - 2.0f * s * c * apq + s * s * aqq;
        float aqq_new = s * s * app + 2.0f * s * c * apq + c * c * aqq;
        a[p*3+p] = app_new;
        a[q*3+q] = aqq_new;
        a[p*3+q] = 0.0f;
        a[q*3+p] = 0.0f;

        // Update the other elements.
        for (int r = 0; r < 3; r++) {
            if (r == p || r == q)
                continue;
            float arp = a[r*3+p];
            float arq = a[r*3+q];
            float arp_new = c * arp - s * arq;
            float arq_new = s * arp + c * arq;
            a[r*3+p] = arp_new;
            a[p*3+r] = arp_new; // symmetry
            a[r*3+q] = arq_new;
            a[q*3+r] = arq_new; // symmetry
        }

        // Update the eigenvector matrix Q.
        for (int r = 0; r < 3; r++) {
            float qrp = Q[r*3+p];
            float qrq = Q[r*3+q];
            Q[r*3+p] = c * qrp - s * qrq;
            Q[r*3+q] = s * qrp + c * qrq;
        }
    }

    // The diagonal of a now holds the eigenvalues.
    eig[0] = a[0];
    eig[1] = a[4];
    eig[2] = a[8];
}


__host__ __device__ inline void svd_decomposition(const float A[9], float U[9], float S[3], float V[9])
{
    // Symmetrize A (row-major): As = 0.5*(A + A^T)
    float As[9];
    As[0] = A[0];
    As[4] = A[4];
    As[8] = A[8];
    As[1] = As[3] = 0.5f * (A[1] + A[3]);
    As[2] = As[6] = 0.5f * (A[2] + A[6]);
    As[5] = As[7] = 0.5f * (A[5] + A[7]);

    // Check near-symmetry
    float diff = 0.0f, normA = 0.0f;
    for (int i = 0; i < 9; ++i) {
        float d = As[i] - A[i];
        diff += d * d;
        normA += A[i] * A[i];
    }
    bool is_symmetric = diff <= 1e-12f * fmaxf(1.0f, normA);

    if (is_symmetric) {
        // Eigen-decompose As (symmetric) → Q columns are eigenvectors, eig are eigenvalues
        float Q[9], eig[3];
        eigen_decomposition(As, Q, eig);

        // Sort eigenvalues/eigenvectors descending
        int idx[3] = {0,1,2};
        for (int i = 0; i < 2; ++i)
            for (int j = i+1; j < 3; ++j)
                if (eig[idx[i]] < eig[idx[j]]) { int t = idx[i]; idx[i] = idx[j]; idx[j] = t; }

        // Build outputs: for symmetric PSD, singular values = eigenvalues
        for (int col = 0; col < 3; ++col) {
            int o = idx[col];
            S[col] = eig[o] > 0.0f ? eig[o] : 0.0f;
            for (int row = 0; row < 3; ++row) {
                float q = Q[row*3 + o];
                U[row*3 + col] = q;
                V[row*3 + col] = q;
            }
        }
        return;
    }

    // Fallback: generic 3x3 SVD via AtA
    float AtA[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < 3; ++k) sum += A[k*3 + i] * A[k*3 + j];
            AtA[i*3 + j] = sum;
        }
    }
    // Enforce exact symmetry
    for (int r = 0; r < 3; ++r)
        for (int c = r+1; c < 3; ++c) {
            float s = 0.5f * (AtA[r*3 + c] + AtA[c*3 + r]);
            AtA[r*3 + c] = s; AtA[c*3 + r] = s;
        }

    float Vtemp[9], eigvals[3];
    eigen_decomposition(AtA, Vtemp, eigvals);

    int idx2[3] = {0,1,2};
    for (int i = 0; i < 2; ++i)
        for (int j = i+1; j < 3; ++j)
            if (eigvals[idx2[i]] < eigvals[idx2[j]]) { int t = idx2[i]; idx2[i] = idx2[j]; idx2[j] = t; }

    float Vsorted[9];
    for (int col = 0; col < 3; ++col) {
        int o = idx2[col];
        S[col] = eigvals[o] > 0.0f ? sqrtf(eigvals[o]) : 0.0f;
        for (int row = 0; row < 3; ++row)
            Vsorted[row*3 + col] = Vtemp[row*3 + o];
    }

    // U = A * V / S (robustly)
    for (int j = 0; j < 3; ++j) {
        float sigma = S[j];
        if (sigma > 1e-8f) {
            for (int i = 0; i < 3; ++i) {
                float s = 0.0f;
                for (int k = 0; k < 3; ++k)
                    s += A[i*3 + k] * Vsorted[k*3 + j];
                U[i*3 + j] = s / sigma;
            }
        } else {
            // degenerate: copy V column
            for (int i = 0; i < 3; ++i)
                U[i*3 + j] = Vsorted[i*3 + j];
        }
        // normalize U column
        float nrm = sqrtf(U[0*3 + j]*U[0*3 + j] + U[1*3 + j]*U[1*3 + j] + U[2*3 + j]*U[2*3 + j]);
        if (nrm > 1e-12f) { U[0*3 + j]/=nrm; U[1*3 + j]/=nrm; U[2*3 + j]/=nrm; }
    }

    // Make dot(U_col, V_col) positive
    for (int j = 0; j < 3; ++j) {
        float dot = U[0*3+j]*Vsorted[0*3+j] + U[1*3+j]*Vsorted[1*3+j] + U[2*3+j]*Vsorted[2*3+j];
        if (dot < 0.0f) { U[0*3+j]*=-1; U[1*3+j]*=-1; U[2*3+j]*=-1; Vsorted[0*3+j]*=-1; Vsorted[1*3+j]*=-1; Vsorted[2*3+j]*=-1; }
    }
    for (int i = 0; i < 9; ++i) V[i] = Vsorted[i];
}


#endif // CUGICP_MATH_CUH
