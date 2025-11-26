#include "fast_gicp/cuda/cutree.cuh"
#include "fast_gicp/cuda/cugicp_core.cuh"
#include "fast_gicp/cuda/matrix.cuh"
#include "fast_gicp/cuda/cugicp_kernels.cuh"
#include "cukd/knn.h"
#include "cukd/builder.h"

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <iomanip>

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

namespace fast_gicp 
{

// ======================================================
//                     CORE FUNCTIONS
// ======================================================

// Constructor: Initialize persistent pointers.
CUGICP::CUGICP() {
    // Only increase the printf FIFO size if the current limit is smaller.
    size_t currentLimit = 0;
    const size_t desiredLimit = 10 * 1024 * 1024;
    cudaError_t err = cudaDeviceGetLimit(&currentLimit, cudaLimitPrintfFifoSize);
    if (err != cudaSuccess) {
        // If querying failed, attempt to set the desired limit.
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, desiredLimit));
    } else if (currentLimit < desiredLimit) {
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, desiredLimit));
    }
}

// Destructor: Free persistent GPU memory.
CUGICP::~CUGICP() {}


bool CUGICP::step_lm(
    float4* d_source,
    float* d_source_covs,
    int source_size,
    std::shared_ptr<CuTree> target_tree,
    float* d_target_covs,
    float* x0_transform,
    float* delta_transform) {

    // Allocate GPU memory
    int *d_correspondences;
    float *d_sq_distances, *d_mahalanobis;
    CUDA_CHECK(cudaMalloc(&d_correspondences, source_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sq_distances, source_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mahalanobis, source_size * 16 * sizeof(float)));
        
    float *d_H, *d_b, *d_error;
    CUDA_CHECK(cudaMalloc(&d_H, 36 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_error, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_H, 0, 36 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_b, 0, 6 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_error, 0, sizeof(float)));

    float *d_x0;
    CUDA_CHECK(cudaMalloc(&d_x0, 16 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x0, x0_transform, 16 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    int threadsPerBlock = THREADS_PER_BLOCK;
    int numBlocks = (source_size + threadsPerBlock - 1) / threadsPerBlock;

    // Step 1: Compute correspondences
    update_correspondences_kernel<<<numBlocks, threadsPerBlock>>>(
        d_source, source_size, d_source_covs, target_tree->tree, d_target_covs,
        d_x0, d_correspondences, d_sq_distances, d_mahalanobis, corr_dist_threshold_);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: Compute Hessian, gradient and error
    linearize_kernel<<<numBlocks, threadsPerBlock>>>(
        d_source, source_size, target_tree->tree, d_correspondences, d_mahalanobis,
        d_x0, d_H, d_b, d_error);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results from GPU
    float H_host[6][6], b_host[6], sum_errors;
    cudaMemcpy(&sum_errors, d_error, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(H_host, d_H, 36 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b_host, d_b, 6 * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free allocated GPU memory
    cudaFree(d_H);
    cudaFree(d_b);
    cudaFree(d_error);
    cudaFree(d_x0);

    //  === Levenberg-Marquardt Initialization ===
    if (lm_lambda_ < 0.0) {
        float max_diag_H = 0.0f;
        for (int d = 0; d < 6; d++) {
            float abs_H_dd = std::abs(H_host[d][d]);
            if (max_diag_H < abs_H_dd)
                max_diag_H = abs_H_dd;
        }
        lm_lambda_ = lm_init_lambda_factor_ * max_diag_H;
    }

    float nu = 2.0f;
    float y0 = sum_errors;
    float yi = 0.0f;
    
    // Solve (H + λI)d = -b using templated LDLT for 6x6 matrices.
    for (int i = 0; i < max_lm_iterations_; i++) {
        float H_lambda[6][6];
        memcpy(H_lambda, H_host, 36 * sizeof(float));
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int j = 0; j < 6; j++) {
            H_lambda[j][j] += lm_lambda_;
        }
        float d[6];
        solveLDLT<6>(H_lambda, b_host, d);

        // Compute transformation update
        float rotation[9];
        so3_exp(d, rotation);
        
        delta_transform[0] = rotation[0]; delta_transform[1] = rotation[1]; delta_transform[2] = rotation[2];  delta_transform[3] = d[3];
        delta_transform[4] = rotation[3]; delta_transform[5] = rotation[4]; delta_transform[6] = rotation[5];  delta_transform[7] = d[4];
        delta_transform[8] = rotation[6]; delta_transform[9] = rotation[7]; delta_transform[10] = rotation[8]; delta_transform[11] = d[5];
        delta_transform[12] = 0.0f;       delta_transform[13] = 0.0f;       delta_transform[14] = 0.0f;        delta_transform[15] = 1.0f;

        float xi_transform[16];
        // Replace matMul4x4 with templated matMul for 4x4 matrices.
        matMul<4>(delta_transform, x0_transform, xi_transform);
        converged_ = is_converged(delta_transform);

        // Compute new error value
        float *d_xi;
        CUDA_CHECK(cudaMalloc(&d_xi, 16 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_xi, xi_transform, 16 * sizeof(float), cudaMemcpyHostToDevice));

        float *d_sum_errors;
        CUDA_CHECK(cudaMalloc(&d_sum_errors, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_sum_errors, 0, sizeof(float)));
        compute_error_kernel<<<numBlocks, threadsPerBlock>>>(
            d_source, source_size, target_tree->tree,
            d_correspondences, d_mahalanobis, d_xi, d_sum_errors);
        CUDA_CHECK(cudaMemcpy(&yi, d_sum_errors, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_sum_errors));
        CUDA_CHECK(cudaFree(d_xi));

        double denominator = 0.0f;
        for (int j = 0; j < 6; j++) {
            denominator += d[j] * (lm_lambda_ * d[j] - b_host[j]);
        }
        if (denominator == 0.0f) return 0.0f;
        double rho = (y0 - yi) / denominator;

        if (DEBUG) {
            if (i == 0) {
                std::cout << std::fixed << std::setprecision(6);
                std::cout << "--- LM optimization (CUDA) ---\n";
                std::cout << std::setw(5) << "i"
                        << std::setw(15) << "y0"
                        << std::setw(15) << "yi"
                        << std::setw(15) << "rho"
                        << std::setw(15) << "lambda"
                        << std::setw(15) << "|delta|"
                        << std::setw(5) << "dec" << std::endl;
            }
        
            float delta_norm = std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2] + d[3]*d[3] + d[4]*d[4] + d[5]*d[5]);
        
            char dec = (rho > 0.0f) ? 'x' : ' ';
            std::cout << std::setw(5) << i
                    << std::setw(15) << y0
                    << std::setw(15) << yi
                    << std::setw(15) << rho
                    << std::setw(15) << lm_lambda_
                    << std::setw(15) << delta_norm
                    << std::setw(5) << dec << std::endl;
        }

        // removed debug trace

        if (rho < 0) {
            if (is_converged(delta_transform)) {
                CUDA_CHECK(cudaFree(d_correspondences));
                CUDA_CHECK(cudaFree(d_sq_distances));
                CUDA_CHECK(cudaFree(d_mahalanobis));
                return true;
            }
            lm_lambda_ = nu * lm_lambda_;
            nu = 2 * nu;
            continue;
        }
        
        lm_lambda_ = lm_lambda_ * std::max(1.0 / 3.0, 1 - std::pow(2 * rho - 1, 3));
        memcpy(x0_transform, xi_transform, 16 * sizeof(float));
        CUDA_CHECK(cudaDeviceSynchronize());
        // TODO: Set final hessian and final Mahalanobis

        CUDA_CHECK(cudaFree(d_correspondences));
        CUDA_CHECK(cudaFree(d_sq_distances));
        CUDA_CHECK(cudaFree(d_mahalanobis));
        return true;
    }

    CUDA_CHECK(cudaFree(d_correspondences));
    CUDA_CHECK(cudaFree(d_sq_distances));
    CUDA_CHECK(cudaFree(d_mahalanobis));
    return false;
}



void CUGICP::set_correspondence_randomness(int k) {
    constexpr int MAX_POWER_OF_2 = 64;
    int clamped_k = 1;
    while (clamped_k * 2 <= k && clamped_k * 2 <= MAX_POWER_OF_2) {
        clamped_k *= 2;
    }
    // if (clamped_k != k) {
    //     printf("Clamping k correspondences from %i to %i\n", k, clamped_k);
    // }
    this->k_correspondences_ = clamped_k;
}

void CUGICP::set_max_correspondence_distance(double max_corr_dist) {
    this->corr_dist_threshold_ = max_corr_dist;
}

void CUGICP::set_maximum_iterations(int max_iterations) {
    this->max_iterations_ = max_iterations;
}

void CUGICP::set_transformation_epsilon(double trans_eps) {
    this->translation_epsilon_ = static_cast<float>(trans_eps);
}

void CUGICP::set_rotation_epsilon(double rot_eps) {
    this->rotation_epsilon_ = static_cast<float>(rot_eps);
}

void CUGICP::set_regularization_method(RegularizationMethod regularization_method) {
    // TODO: Implement the regularization method
    this->regularization_ = regularization_method;
}

void CUGICP::set_neighbors_distance(double neigh_dist) {
    this->neighbors_distance_ = neigh_dist;
}

const float* CUGICP::get_final_transformation() const {
    return final_transformation_;
}

int CUGICP::get_num_iterations() const {
    return num_iterations_;
}

bool CUGICP::has_converged() const {
    return converged_;
}

float* CUGICP::compute_covariances(float4* d_cloud, int cloud_size, std::shared_ptr<CuTree> tree) {

    std::shared_ptr<CuTree> local_tree = tree;
    if(!tree) {
        local_tree = compute_cutree(d_cloud, cloud_size);
    }

    float* d_covariances;
    CUDA_CHECK(cudaMalloc(&d_covariances, cloud_size * 16 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_covariances, 0, cloud_size * 16 * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());

#define LAUNCH_KERNEL(K) \
{ \
    compute_covariances_kernel<K><<<numBlocks, threadsPerBlock>>>( \
        d_cloud, cloud_size, local_tree->tree, \
        neighbors_distance_, d_covariances, regularization_); \
}

    int threadsPerBlock = k_correspondences_;
    int numBlocks = cloud_size;
    switch (k_correspondences_) {
        case 1:  LAUNCH_KERNEL(1); break;
        case 2:  LAUNCH_KERNEL(2); break;
        case 4:  LAUNCH_KERNEL(4); break;
        case 8:  LAUNCH_KERNEL(8); break;
        case 16: LAUNCH_KERNEL(16); break;
        case 32: LAUNCH_KERNEL(32); break;
        case 64: LAUNCH_KERNEL(64); break;
        default: 
            std::cerr << "Error: Invalid k_correspondences_ value\n"; 
            return nullptr;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if(!tree)
        free_cutree(local_tree);

    return d_covariances;
}

void CUGICP::cutree_to_cpu(std::shared_ptr<CuTree>& hostTree, const std::shared_ptr<CuTree> deviceTree) {
    if(!deviceTree)
        return;

    hostTree.reset(new CuTree);

    // 1. Copy scalar fields.
    hostTree->tree.bounds   = deviceTree->tree.bounds;
    hostTree->tree.numPrims = deviceTree->tree.numPrims;
    hostTree->tree.numNodes = deviceTree->tree.numNodes;
    
    // 2. Allocate memory for the node array and copy the nodes.
    size_t nodesSize = hostTree->tree.numNodes * sizeof(typename decltype(hostTree->tree)::Node);
    hostTree->tree.nodes = static_cast<typename decltype(hostTree->tree)::Node*>(malloc(nodesSize));
    CUDA_CHECK(cudaMemcpy(hostTree->tree.nodes, deviceTree->tree.nodes, nodesSize, cudaMemcpyDeviceToHost));
    
    // 3. Allocate memory for the primitive IDs and copy them.
    size_t primIDsSize = hostTree->tree.numPrims * sizeof(uint32_t);
    hostTree->tree.primIDs = static_cast<uint32_t*>(malloc(primIDsSize));
    CUDA_CHECK(cudaMemcpy(hostTree->tree.primIDs, deviceTree->tree.primIDs, primIDsSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaDeviceSynchronize());
}

void CUGICP::cutree_to_gpu(std::shared_ptr<CuTree>& deviceTree, const std::shared_ptr<CuTree> hostTree, const float4* dataPointer) {
    if(!hostTree)
        return;

    deviceTree.reset(new CuTree);

    // 1. Copy scalar fields.
    deviceTree->tree.bounds   = hostTree->tree.bounds;
    deviceTree->tree.numPrims = hostTree->tree.numPrims;
    deviceTree->tree.numNodes = hostTree->tree.numNodes;
    
    // 2. Allocate memory for the node array and copy the nodes.
    size_t nodesSize = deviceTree->tree.numNodes * sizeof(typename decltype(deviceTree->tree)::Node);
    CUDA_CHECK(cudaMalloc(&(deviceTree->tree.nodes), nodesSize));
    CUDA_CHECK(cudaMemcpy(deviceTree->tree.nodes, hostTree->tree.nodes, nodesSize, cudaMemcpyHostToDevice));
    
    // 3. Allocate memory for the primitive IDs and copy them.
    size_t primIDsSize = deviceTree->tree.numPrims * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(&(deviceTree->tree.primIDs), primIDsSize));
    CUDA_CHECK(cudaMemcpy(deviceTree->tree.primIDs, hostTree->tree.primIDs, primIDsSize, cudaMemcpyHostToDevice));
    
    // 4. Refill the data array pointer with the given pointer.
    deviceTree->tree.data = dataPointer;

    CUDA_CHECK(cudaDeviceSynchronize());
}

std::shared_ptr<CuTree> CUGICP::compute_cutree(float4* d_cloud, int cloud_size) {
    std::shared_ptr<CuTree> cutree(new CuTree);
    cukd::BuildConfig buildConfig{};
    cukd::buildTree(cutree->tree, d_cloud, cloud_size, buildConfig);
    CUDA_CHECK(cudaDeviceSynchronize());
    return cutree;
}

void CUGICP::free_cutree(std::shared_ptr<CuTree> cutree) {
    if (!cutree)
        return;
    
        cukd::free(cutree->tree);
}

bool CUGICP::is_converged(const float delta[16]) {
    float R[9] = {
        delta[0], delta[1], delta[2],
        delta[4], delta[5], delta[6],
        delta[8], delta[9], delta[10]
    };

    float R_diff[9] = {
        R[0] - 1.0f, R[1],        R[2],
        R[3],        R[4] - 1.0f, R[5],
        R[6],        R[7],        R[8] - 1.0f
    };

    float t[3] = { delta[3], delta[7], delta[11] };

    float max_r_delta = 0.0f;
    for (int i = 0; i < 9; i++) {
        max_r_delta = fmaxf(max_r_delta, fabsf(R_diff[i]) / rotation_epsilon_);
    }
    float max_t_delta = 0.0f;
    for (int i = 0; i < 3; i++) {
        max_t_delta = fmaxf(max_t_delta, fabsf(t[i]) / translation_epsilon_);
    }
    return fmaxf(max_r_delta, max_t_delta) < 1.0f;
}


void CUGICP::align_cuda(
    float4* d_source,
    float* d_source_covs,
    int source_size,
    std::shared_ptr<CuTree> target_tree,
    float* d_target_covs,
    float* prior) {    
    if(!(d_source && d_source_covs && source_size > 0 &&
         target_tree && d_target_covs))
    {
        printf("CUDA Alignment could not start, check if all the parameters have been set!\n");
        printf("%p, %p, %d, %p, %p\n", d_source, d_source_covs, source_size, target_tree.get(), d_target_covs);
        return;
    }

    float x0_transform[16];
    memcpy(x0_transform, prior, 16 * sizeof(float));

    num_iterations_ = 0;
    converged_ = 0;
    lm_lambda_ = -1.0;
    std::memset(final_transformation_, 0, 16 * sizeof(float));
    final_transformation_[0] = 1.f;
    final_transformation_[5] = 1.f;
    final_transformation_[10] = 1.f;
    final_transformation_[15] = 1.f;
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < max_iterations_ && !converged_; i++) {
        num_iterations_++;
        float delta_transform[16];
        if (!step_lm(
            d_source,
            d_source_covs,
            source_size,
            target_tree,
            d_target_covs,
            x0_transform, 
            delta_transform)) {
            fprintf(stderr, "LM did not converge!!\n");
            break;
        }
        converged_ = is_converged(delta_transform);
    }

    memcpy(final_transformation_, x0_transform, 16 * sizeof(float));
    CUDA_CHECK(cudaDeviceSynchronize());
}

}
