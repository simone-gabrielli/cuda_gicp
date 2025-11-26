#ifndef CUGICP_CUH
#define CUGICP_CUH

#include <cuda_runtime.h>
#include <memory>
#include <fast_gicp/gicp/gicp_settings.hpp>

namespace fast_gicp
{

#define CUDA_CHECK( call )                                         \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if (rc != cudaSuccess) {                                            \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      throw std::runtime_error("fatal cuda error");                     \
    }                                                                   \
  }


class CuTree; // Forward Declaration

/**
 * @brief CUDA-based Generalized ICP (GICP) alignment engine.
 *
 * This class provides a Levenberg-Marquardt-based point cloud registration method 
 * using GPU-accelerated covariance computation and optimization.
 */
class CUGICP {
public:

    // ===================
    //     Core Methods
    // ===================

    /**
     * @brief Constructor. Initializes parameters and clears internal state.
     */
    CUGICP();

    /**
     * @brief Destructor. Frees any allocated GPU memory.
     */
    ~CUGICP();

    /**
     * @brief Runs the full CUDA-based GICP alignment.
     *
     * @param d_source Source cloud in device memory.
     * @param d_source_covs Covariances for the source cloud (device memory).
     * @param source_size Number of source points.
     * @param target_tree KD-tree for target cloud.
     * @param d_target_covs Covariances for the target cloud (device memory).
     * @param prior Initial transformation (4x4 row-major).
     */
    void align_cuda(
        float4* d_source,
        float* d_source_covs,
        int source_size,
        std::shared_ptr<CuTree> target_tree,
        float* d_target_covs,
        float* prior);

    /**
     * @brief Computes per-point covariance matrices on the GPU using k-NN.
     *
     * @param d_cloud Query cloud on the GPU.
     * @param cloud_size Number of query points.
     * @param tree Optional precomputed CuTree for neighbor searches. If nullptr, a new tree is built from d_cloud.
     * @return Pointer to device memory containing N 4x4 covariance matrices (caller must cudaFree).
     */
    float* compute_covariances(
        float4* d_cloud, 
        int cloud_size,
        std::shared_ptr<CuTree> tree = nullptr);

  /**
   * @brief Computes normals from 4x4 covariances and also returns singular values (s0,s2)
   * for each point. The svals array must be of length 2*N and will contain [s0,s2] per-point.
   */
  void compute_normals_from_covariances_and_svals(
    const float* d_covs,
    float3* d_normals,
    float* d_svals,
    int N);

  /**
   * @brief Classify points using normals and precomputed singular values. The svals array
   * is expected to be packed as [s0,s2,s0,s2,...].
   */
  void classify_normals(
    const float4* d_points,
    const float3* d_normals,
    const float* d_svals,
    uint8_t* d_types,
    int N,
    float max_angle,
    const float3 range,
    float planarity_thresh = 0.1f);

        
    /**
     * @brief Builds a GPU KD-tree (CuTree) for fast spatial queries.
     *
     * @param d_cloud Cloud to index (in device memory).
     * @param cloud_size Number of points.
     * @return Shared pointer to constructed CuTree (caller must call free_cutree).
     */
    std::shared_ptr<CuTree> compute_cutree(
        float4* d_cloud, 
        int cloud_size);

    // ===================
    //       Setters
    // ===================

    /**
     * @brief Sets the number of neighbors used in k-NN covariance estimation.
     *
     * Value is clamped to a power of 2 (up to 64) for performance reasons.
     *
     * @param k Number of neighbors.
     */
    void set_correspondence_randomness(int k);

    /**
     * @brief Sets the maximum allowed distance for point correspondences.
     *
     * @param max_corr_dist Maximum Euclidean distance threshold.
     */
    void set_max_correspondence_distance(double max_corr_dist);

    /**
     * @brief Sets the number of maximum outer LM iterations.
     *
     * @param max_iterations Number of iterations.
     */
    void set_maximum_iterations(int max_iterations);

    /**
     * @brief Sets the convergence threshold for translation (L2 norm).
     *
     * @param trans_eps Minimum translation step.
     */
    void set_transformation_epsilon(double trans_eps);

    /**
     * @brief Sets the convergence threshold for rotation (matrix delta).
     *
     * @param rot_eps Minimum rotation step.
     */
    void set_rotation_epsilon(double rot_eps);

    /**
     * @brief Sets the regularization method (reserved for future use).
     *
     * @param regularization_method Integer flag for method.
     */
    void set_regularization_method(RegularizationMethod regularization_method);

    /**
     * @brief Sets the maximum distance for neighbor inclusion when computing covariances.
     *
     * @param neigh_dist Maximum radius to consider neighbors.
     */
    void set_neighbors_distance(double neigh_dist);

    // ===================
    //       Getters
    // ===================

    /**
     * @return Pointer to the 4x4 row-major final transformation result.
     */
    const float* get_final_transformation() const;

    /**
     * @return Number of outer LM iterations performed.
     */
    int get_num_iterations() const;

    /**
     * @return True if alignment has converged.
     */
    bool has_converged() const;


    // ===================
    //       Static
    // ===================

    static void cutree_to_cpu(std::shared_ptr<CuTree>& hostTree, const std::shared_ptr<CuTree> deviceTree);

    static void cutree_to_gpu(std::shared_ptr<CuTree>& deviceTree, const std::shared_ptr<CuTree> hostTree, const float4* dataPointer); 

    /**
     * @brief Frees a CuTree and its associated resources.
     * 
     * @param tree Shared pointer to the CuTree.
     */
    static void free_cutree(std::shared_ptr<CuTree> tree);

protected:

    /**
     * @brief Checks convergence based on delta transformation.
     *
     * @param delta The 4x4 delta transformation matrix.
     * @return True if both rotation and translation deltas are below thresholds.
     */
    bool is_converged(const float delta[16]);

    /**
     * @brief Executes a single LM step of GICP optimization.
     *
     * @param d_source Source point cloud.
     * @param d_source_covs Source covariances.
     * @param source_size Number of source points.
     * @param target_tree Target spatial index.
     * @param d_target_covs Target covariances.
     * @param x0_transform Current transformation.
     * @param delta_transform Output update matrix.
     * @return True if step succeeds (update is valid).
     */
    bool step_lm(
        float4* d_source,
        float* d_source_covs,
        int source_size,
        std::shared_ptr<CuTree> target_tree,
        float* d_target_covs,
        float* x0_transform,
        float* delta_transform);

protected:

    // ===================
    //   Optimization Params
    // ===================

    RegularizationMethod regularization_ = RegularizationMethod::NONE; ///< Covariance Regularization Method
    float rotation_epsilon_ = 2e-3f;         ///< Convergence threshold for rotation.
    float translation_epsilon_ = 5e-4f;      ///< Convergence threshold for translation.
    int max_iterations_ = 50;                ///< Max outer LM iterations.
    int k_correspondences_ = 16;             ///< Neighbors for covariance estimation.
    double neighbors_distance_ = 1e9;        ///< Max radius for neighbor search.
    double corr_dist_threshold_ = 1e9;       ///< Max correspondence distance.

    int max_lm_iterations_ = 10;             ///< Max inner LM iterations.
    float lm_init_lambda_factor_ = 1e-9f;    ///< Initial damping for LM.

    // ===================
    //      State
    // ===================

    float final_transformation_[16];         ///< Output: final transformation matrix.
    int num_iterations_ = 0;                 ///< Number of optimization steps run.
    bool converged_ = false;                 ///< Whether convergence was reached.
    float lm_lambda_ = -1.0f;                ///< LM damping factor.
};

}

#endif // CUGICP_CUH
