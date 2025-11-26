#ifndef CUGICP_KERNELS_CUH
#define CUGICP_KERNELS_CUH

#define DEBUG false

#include "cukd/knn.h"
#include "cukd/builder.h"
#include "fast_gicp/cuda/matrix.cuh"

#include "fast_gicp/gicp/gicp_settings.hpp"

namespace fast_gicp 
{

template <int K, typename PointT = float4, typename PointTraits = cukd::default_data_traits<float4>>
static __global__ void compute_covariances_kernel(
    const float4* __restrict__ d_queries, int numQueries,
    const cukd::SpatialKDTree<PointT, PointTraits> tree,
    float radius,
    float* d_covariances,
    fast_gicp::RegularizationMethod regularization_method)
{
    int query_id = blockIdx.x;
    int neighbor_id = threadIdx.x;

    if (query_id >= numQueries || neighbor_id >= K) return;

    __shared__ float4 s_neighbors[K];
    __shared__ float4 s_centroid;
    __shared__ float cov[4][4];
    __shared__ int s_indices[K];
    __shared__ int valid_neighbors;

    if (threadIdx.x == 0) {
        cukd::HeapCandidateList<K> result(radius);
        cukd::stackBased::knn<decltype(result), PointT, PointTraits>(
            result, tree, d_queries[query_id]);

        valid_neighbors = 0;
        for (int i = 0; i < K; i++) {
            s_indices[i] = result.get_pointID(i);
            if (s_indices[i] >= 0)
                valid_neighbors++;
        }
    }
    __syncthreads();

    int ID = s_indices[neighbor_id];
    s_neighbors[neighbor_id] = (ID < 0) ? make_float4(0.f, 0.f, 0.f, 0.f) : tree.data[ID];
    __syncthreads();

    if (threadIdx.x == 0) {
        s_centroid = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                cov[i][j] = 0.f;
    }
    __syncthreads();

    if (ID >= 0 && valid_neighbors > 0) {
        atomicAdd(&s_centroid.x, s_neighbors[neighbor_id].x / valid_neighbors);
        atomicAdd(&s_centroid.y, s_neighbors[neighbor_id].y / valid_neighbors);
        atomicAdd(&s_centroid.z, s_neighbors[neighbor_id].z / valid_neighbors);
        atomicAdd(&s_centroid.w, s_neighbors[neighbor_id].w / valid_neighbors);
    }
    __syncthreads();

    if (ID >= 0) {
        float4 diff = make_float4(
            s_neighbors[neighbor_id].x - s_centroid.x,
            s_neighbors[neighbor_id].y - s_centroid.y,
            s_neighbors[neighbor_id].z - s_centroid.z,
            s_neighbors[neighbor_id].w - s_centroid.w
        );

        atomicAdd(&cov[0][0], diff.x * diff.x);
        atomicAdd(&cov[0][1], diff.x * diff.y);
        atomicAdd(&cov[0][2], diff.x * diff.z);
        atomicAdd(&cov[0][3], diff.x * diff.w);

        atomicAdd(&cov[1][0], diff.y * diff.x);
        atomicAdd(&cov[1][1], diff.y * diff.y);
        atomicAdd(&cov[1][2], diff.y * diff.z);
        atomicAdd(&cov[1][3], diff.y * diff.w);

        atomicAdd(&cov[2][0], diff.z * diff.x);
        atomicAdd(&cov[2][1], diff.z * diff.y);
        atomicAdd(&cov[2][2], diff.z * diff.z);
        atomicAdd(&cov[2][3], diff.z * diff.w);

        atomicAdd(&cov[3][0], diff.w * diff.x);
        atomicAdd(&cov[3][1], diff.w * diff.y);
        atomicAdd(&cov[3][2], diff.w * diff.z);
        atomicAdd(&cov[3][3], diff.w * diff.w);
    }
    __syncthreads();

    int baseIdx = query_id * 16;
    if (threadIdx.x == 0) {
        for (int i = 0; i < 16; i++) d_covariances[baseIdx + i] = 0.f;

        if (valid_neighbors > 3) {
            // Step 1: Normalize the top-left 3x3 block
            float Cov3[9];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    Cov3[i * 3 + j] = cov[i][j] / valid_neighbors;

            float reg[9];

            if (regularization_method == fast_gicp::RegularizationMethod::NONE) {
                for (int i = 0; i < 9; i++) reg[i] = Cov3[i];
            }
            if (regularization_method == fast_gicp::RegularizationMethod::FROBENIUS) {
                float C[9], C_inv[9];
                for (int i = 0; i < 9; i++) C[i] = Cov3[i];
                C[0] += 1e-3f; C[4] += 1e-3f; C[8] += 1e-3f;
                invertMatrix<3>(C, C_inv);
                float norm = matFrobeniusNorm<3>(C_inv);
                for (int i = 0; i < 9; i++) reg[i] = norm * C[i];
            }
            else {
                // Compute SVD of the 3x3 covariance matrix Cov3.
                float U[9], S[3], V[9];
                
                svd_decomposition(Cov3, U, S, V);

                // Regularize singular values into the vector "values".
                float values[3];
                if (regularization_method == fast_gicp::RegularizationMethod::PLANE) {
                    values[0] = 1.f; values[1] = 1.f; values[2] = 1e-3f;
                } else if (regularization_method == fast_gicp::RegularizationMethod::MIN_EIG) {
                    values[0] = fmaxf(S[0], 1e-3f);
                    values[1] = fmaxf(S[1], 1e-3f);
                    values[2] = fmaxf(S[2], 1e-3f);
                } else if (regularization_method == fast_gicp::RegularizationMethod::NORMALIZED_MIN_EIG) {
                    float maxS = S[0]; // S is in descending order.
                    values[0] = fmaxf(S[0] / maxS, 1e-3f);
                    values[1] = fmaxf(S[1] / maxS, 1e-3f);
                    values[2] = fmaxf(S[2] / maxS, 1e-3f);
                }

                // Build a 3x3 diagonal matrix D from "values".
                float D[9] = {0.f};
                D[0] = values[0];
                D[4] = values[1];
                D[8] = values[2];

                // Compute intermediate product: M = U * D.
                float M[9];
                matMul<3>(U, D, M);

                // Compute V^T by transposing V.
                float Vt[9];
                matTranspose<3>(V, Vt);

                // Finally, compute the regularized covariance: reg = M * V^T.
                matMul<3>(M, Vt, reg);

                // Enforce exact symmetry on reg (numerical stability)
                for (int r = 0; r < 3; ++r) {
                    for (int c = r+1; c < 3; ++c) {
                        float s = 0.5f * (reg[r*3 + c] + reg[c*3 + r]);
                        reg[r*3 + c] = s;
                        reg[c*3 + r] = s;
                    }
                }
            }

            // Step 2: Fill the output 4x4 matrix (row-major)
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    d_covariances[baseIdx + i * 4 + j] = reg[i * 3 + j];
        } else {
            d_covariances[baseIdx + 15] = COV_33_INVALID;
        }
    }
}

static __global__ void compute_normals_from_covariances_kernel(
    const float* __restrict__ d_covs, // size: N * 16 (4x4 matrices, row-major)
    float3* d_normals,                // output normals (unit length)
    int N)                                   // number of matrices
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float* cov4x4 = &d_covs[idx * 16];

    float3 normal;
    if(cov4x4[15] != COV_33_INVALID)
    {
        // Extract the top-left 3x3 submatrix (row-major)
        float cov3x3[9] = {
            cov4x4[0], cov4x4[1], cov4x4[2],
            cov4x4[4], cov4x4[5], cov4x4[6],
            cov4x4[8], cov4x4[9], cov4x4[10]
        };

        float U[9], S[3], V[9];
        svd_decomposition(cov3x3, U, S, V);

        // The smallest singular value is S[2] (singular values are sorted descending)
        // and the corresponding eigenvector is stored in the third column of V.
        // V is stored in row-major order, so the third column is at indices 6, 7, 8.
        normal.x = V[6];
        normal.y = V[7];
        normal.z = V[8];

        // Normalize to ensure unit length
        float len = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        if (len > 1e-6f) {
            normal.x /= len;
            normal.y /= len;
            normal.z /= len;
        } else {
            // Fallback normal if degenerate
            normal = make_float3(0.0f, 0.0f, 0.0f);
        }
    } else {
        // Fallback normal if degenerate
        normal = make_float3(0.0f, 0.0f, 0.0f);
    }

    d_normals[idx] = normal;
}

template <typename PointT = float4, typename PointTraits = cukd::default_data_traits<float4>>
static __global__ void update_correspondences_kernel(
    const float4* __restrict__ d_source,
    int source_size,
    const float* __restrict__ source_covs,
    const cukd::SpatialKDTree<PointT, PointTraits> target_tree,
    const float* __restrict__ target_covs,
    const float* __restrict__ d_transform,
    int* __restrict__ correspondences,
    float* __restrict__ sq_distances,
    float* __restrict__ mahalanobis,
    float corr_dist_threshold) {

    __shared__ float shared_transform[16];
    if (threadIdx.x < 16) {
        shared_transform[threadIdx.x] = d_transform[threadIdx.x];
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= source_size) return;

    float4 pt = d_source[tid];

    // Transform source point
    float4 transformed_pt;
    transformed_pt.x = shared_transform[0] * pt.x + shared_transform[1] * pt.y + shared_transform[2] * pt.z + shared_transform[3];
    transformed_pt.y = shared_transform[4] * pt.x + shared_transform[5] * pt.y + shared_transform[6] * pt.z + shared_transform[7];
    transformed_pt.z = shared_transform[8] * pt.x + shared_transform[9] * pt.y + shared_transform[10] * pt.z + shared_transform[11];
    transformed_pt.w = 1.0f;

    // KNN search
    cukd::FcpSearchParams params;
    params.cutOffRadius = corr_dist_threshold;
    int nearest_idx = cukd::stackBased::fcp(target_tree, transformed_pt, params);

    // Invalid nearest point
    if (nearest_idx < 0 || nearest_idx >= target_tree.numPrims) {
        correspondences[tid] = -1;
        return;
    }

    // Invalid covariances
    if (source_covs[tid * 16 + 15] == COV_33_INVALID || target_covs[nearest_idx * 16 + 15] == COV_33_INVALID) {
        correspondences[tid] = -1;
        return;
    }

    float4 closest = target_tree.data[nearest_idx];
    float dx = transformed_pt.x - closest.x;
    float dy = transformed_pt.y - closest.y;
    float dz = transformed_pt.z - closest.z;
    float sq_dist = dx * dx + dy * dy + dz * dz;

    if (sq_dist >= corr_dist_threshold * corr_dist_threshold) {
        // if (DEBUG) {
        //     printf("[DEBUG] sq_dist %.6f exceeds threshold %.6f\n", sq_dist, corr_dist_threshold * corr_dist_threshold);
        // }
        correspondences[tid] = -1;
        return;
    }

    correspondences[tid] = nearest_idx;
    sq_distances[tid] = sq_dist;

    const float* cov_A = &source_covs[tid * 16];
    const float* cov_B = &target_covs[nearest_idx * 16];

    // Compute combined covariance RCR = cov_B + T * cov_A * T^T
    float M1[16], M2[16], transform_t[16];
    // Transpose shared_transform into transform_t
    for (int k = 0; k < 4; ++k) {
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            transform_t[l * 4 + k] = shared_transform[k * 4 + l];
        }
    }

    // Use templated 4x4 multiplication.
    matMul<4>(shared_transform, cov_A, M1);
    matMul<4>(M1, transform_t, M2);

    float RCR[16];
    for (int i = 0; i < 16; ++i) {
        RCR[i] = cov_B[i] + M2[i];
    }
    RCR[15] = 1.0f;

    float RCR_inv[16];
    bool success = invertMatrix<4>(RCR, RCR_inv);
    if (!success) {
        correspondences[tid] = -1;
        return;
    }

    for (int i = 0; i < 16; ++i) {
        mahalanobis[tid * 16 + i] = RCR_inv[i];
    }
    mahalanobis[tid * 16 + 15] = 0.0f;

    // if (DEBUG && tid == 5000) {
    //     printf("[SUCCESS] Correspondence %d → %d, sq_dist=%.6f\n", tid, nearest_idx, sq_dist);
    // }
}

template <typename PointT = float4, typename PointTraits = cukd::default_data_traits<float4>>
static __global__ void linearize_kernel(
    const float4* __restrict__ d_source,
    const int source_size,
    const cukd::SpatialKDTree<PointT, PointTraits> d_target,
    const int* __restrict__ d_correspondences,
    const float* __restrict__ d_mahalanobis,
    const float* __restrict__ d_transform,
    float* __restrict__ d_H,
    float* __restrict__ d_b,
    float* __restrict__ d_error) {
        
    __shared__ float shared_transform[16];
    if (threadIdx.x < 16)
        shared_transform[threadIdx.x] = d_transform[threadIdx.x];
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= source_size) return;

    int target_idx = d_correspondences[tid];
    if (target_idx < 0) return;

    float4 source_pt = d_source[tid];
    PointT target_pt = d_target.data[target_idx];

    // Transform source point
    float4 transed_pt;
    transed_pt.x = shared_transform[0] * source_pt.x + shared_transform[1] * source_pt.y + shared_transform[2] * source_pt.z + shared_transform[3];
    transed_pt.y = shared_transform[4] * source_pt.x + shared_transform[5] * source_pt.y + shared_transform[6] * source_pt.z + shared_transform[7];
    transed_pt.z = shared_transform[8] * source_pt.x + shared_transform[9] * source_pt.y + shared_transform[10] * source_pt.z + shared_transform[11];
    transed_pt.w = 1.0f;

    // Compute error vector (target - transformed source)
    float4 error;
    error.x = target_pt.x - transed_pt.x;
    error.y = target_pt.y - transed_pt.y;
    error.z = target_pt.z - transed_pt.z;
    error.w = 0.0f;

    // if(DEBUG)
    // {
    //     if (tid == 5000) {
    //         printf("[linearize_kernel] Thread %d: error vector = (%.6f, %.6f, %.6f)\n", tid, error.x, error.y, error.z);
    //     }
    // }

    float local_error = 0.0f;
    float temp[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 3; i++) {
        #pragma unroll
        for (int j = 0; j < 3; j++) {
            temp[i] += d_mahalanobis[tid * 16 + i * 4 + j] * ((&error.x)[j]);
        }
    }
    for (int i = 0; i < 3; i++) {
        local_error += temp[i] * ((&error.x)[i]);
    }
    atomicAdd(d_error, local_error);

    // Build the 4x6 Jacobian matrix J
    float J[24] = {0.0f};
    float skew[9];
    skewSymmetric(make_float3(transed_pt.x, transed_pt.y, transed_pt.z), skew);

    // Rotation part (using skew symmetric matrix)
    J[0]  = skew[0]; J[1]  = skew[1]; J[2]  = skew[2];
    J[6]  = skew[3]; J[7]  = skew[4]; J[8]  = skew[5];
    J[12] = skew[6]; J[13] = skew[7]; J[14] = skew[8];

    // Translation part
    J[3]  = -1.0f; J[4]  = 0.0f; J[5]  = 0.0f;
    J[9]  =  0.0f; J[10] = -1.0f; J[11] =  0.0f;
    J[15] =  0.0f; J[16] =  0.0f; J[17] = -1.0f;
    // 4th row remains zero

    // Compute JT_M = J^T * (Mahalanobis) where Mahalanobis is 4x4.
    float JT_M[24] = {0.0f}; // 6x4 matrix
    for (int k = 0; k < 6; k++) {
        for (int l = 0; l < 4; l++) {
            #pragma unroll
            for (int m = 0; m < 4; m++) {
                JT_M[k * 4 + l] += J[m * 6 + k] * d_mahalanobis[tid * 16 + m * 4 + l];
            }
        }
    }

    // Compute local Hessian: local_H = JT_M * J (6x6)
    float local_H[36] = {0.0f};
    for (int k = 0; k < 6; k++) {
        for (int l = 0; l < 6; l++) {
            #pragma unroll
            for (int m = 0; m < 4; m++) {
                local_H[k * 6 + l] += JT_M[k * 4 + m] * J[m * 6 + l];
            }
        }
    }
    for (int k = 0; k < 6; k++) {
        for (int l = 0; l < 6; l++) {
            atomicAdd(&d_H[k * 6 + l], local_H[k * 6 + l]);
        }
    }

    // Compute gradient b = J^T * (Mahalanobis * error)
    float local_b[6] = {0.0f};
    for (int k = 0; k < 6; k++) {
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            local_b[k] += JT_M[k * 4 + m] * ((&error.x)[m]);
        }
    }
    for (int k = 0; k < 6; k++) {
        atomicAdd(&d_b[k], local_b[k]);
    }

    // if (DEBUG)
    // {
    //     if (tid == 5000) {
    //         printf("[linearize_kernel] Thread %d: local_H[0] = %.6f, local_b[0] = %.6f\n", tid, local_H[0], local_b[0]);
    //     }
    // }
}

template <typename PointT = float4, typename PointTraits = cukd::default_data_traits<float4>>
static __global__ void compute_error_kernel(
    const float4* __restrict__ d_source,
    const int source_size,
    const cukd::SpatialKDTree<PointT, PointTraits> target_tree,
    const int* __restrict__ d_correspondences,
    const float* __restrict__ d_mahalanobis,
    const float* __restrict__ d_transform,
    float* __restrict__ d_sum_errors)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= source_size) return;

    __shared__ float shared_transform[16];
    if (threadIdx.x < 16) {
        shared_transform[threadIdx.x] = d_transform[threadIdx.x];
    }
    __syncthreads();

    int target_idx = d_correspondences[tid];
    if (target_idx < 0 || target_idx >= target_tree.numPrims) return;

    int mahalanobis_offset = tid * 16 + 12;
    if (mahalanobis_offset >= source_size * 16) return;

    float4 source_pt = d_source[tid];
    PointT target_pt = target_tree.data[target_idx];

    float4 transed_pt;
    transed_pt.x = shared_transform[0] * source_pt.x + shared_transform[1] * source_pt.y + shared_transform[2] * source_pt.z + shared_transform[3];
    transed_pt.y = shared_transform[4] * source_pt.x + shared_transform[5] * source_pt.y + shared_transform[6] * source_pt.z + shared_transform[7];
    transed_pt.z = shared_transform[8] * source_pt.x + shared_transform[9] * source_pt.y + shared_transform[10] * source_pt.z + shared_transform[11];
    transed_pt.w = 1.0f;

    float4 error;
    error.x = target_pt.x - transed_pt.x;
    error.y = target_pt.y - transed_pt.y;
    error.z = target_pt.z - transed_pt.z;
    error.w = 0.0f;

    float temp[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 3; i++) {
        #pragma unroll
        for (int j = 0; j < 3; j++) {
            temp[i] += d_mahalanobis[tid * 16 + i * 4 + j] * ((&error.x)[j]);
        }
    }

    float local_error = 0.0f;
    for (int i = 0; i < 3; i++) {
        local_error += temp[i] * ((&error.x)[i]);
    }

    atomicAdd(d_sum_errors, local_error);

    // if (DEBUG && tid == 5000) {
    //     printf("[compute_error_kernel] error=(%.4f, %.4f, %.4f), local_error=%.6f\n", error.x, error.y, error.z, local_error);
    //     printf("[compute_error_kernel] mahalanobis[0]=%.4f\n", d_mahalanobis[tid * 16]);
    // }
}


}

#endif // CUGICP_KERNELS_CUH