#ifndef FAST_GICP_CUGICP_IMPL_HPP
#define FAST_GICP_CUGICP_IMPL_HPP

#include "fast_gicp/so3/so3.hpp"
#include "fast_gicp/cuda/cugicp_core.cuh"
#include <pcl/io/ply_io.h>

namespace fast_gicp {

template <typename PointSource, typename PointTarget>
FastCUGICP<PointSource, PointTarget>::FastCUGICP() : FastGICP<PointSource, PointTarget>() {
  this->reg_name_ = "FastCUGICP";
  this->cugicp_core_ = std::unique_ptr<CUGICP>(new CUGICP);
}

template <typename PointSource, typename PointTarget>
FastCUGICP<PointSource, PointTarget>::~FastCUGICP() {
  clearSource();
  clearTarget();
}

template <typename PointSource, typename PointTarget>
void FastCUGICP<PointSource, PointTarget>::clearSourceCUDA(){
  source_size_ = 0;
  if (d_source_) { cudaFree(d_source_); d_source_ = nullptr; }
  if (d_source_covs_) { cudaFree(d_source_covs_); d_source_covs_ = nullptr; }
  if (source_tree_) { cugicp_core_->free_cutree(source_tree_); source_tree_ = nullptr; }
}

template <typename PointSource, typename PointTarget>
void FastCUGICP<PointSource, PointTarget>::clearTargetCUDA(){
  target_size_ = 0;
  if (d_target_) { cudaFree(d_target_); d_target_ = nullptr; }
  if (d_target_covs_) { cudaFree(d_target_covs_); d_target_covs_ = nullptr; }
  if (target_tree_) { cugicp_core_->free_cutree(target_tree_); target_tree_ = nullptr; }
}

template <typename PointSource, typename PointTarget>
void FastCUGICP<PointSource, PointTarget>::clearSource() {
  FastGICP<PointSource, PointTarget>::clearSource();
  clearSourceCUDA();
}

template <typename PointSource, typename PointTarget>
void FastCUGICP<PointSource, PointTarget>::clearTarget() {
  FastGICP<PointSource, PointTarget>::clearTarget();
  clearTargetCUDA();
}

template <typename PointSource, typename PointTarget>
void FastCUGICP<PointSource, PointTarget>::setInputSource(float4* source, int source_size) {
  clearSourceCUDA();

  source_size_ = source_size;

  CUDA_CHECK(cudaMalloc(&d_source_, source_size_ * sizeof(float4)));
  CUDA_CHECK(cudaMemcpy(d_source_, source, source_size_ * sizeof(float4), cudaMemcpyHostToDevice));

  source_tree_ = cugicp_core_->compute_cutree(d_source_, source_size_);

  cugicp_core_->set_regularization_method(RegularizationMethod((int)regularization_method_));
  cugicp_core_->set_neighbors_distance(neighbors_distance_);

  d_source_covs_ = cugicp_core_->compute_covariances(d_source_, source_size_, source_tree_);
}

template <typename PointSource, typename PointTarget>
void FastCUGICP<PointSource, PointTarget>::setInputTarget(float4* target, int target_size) {
  clearTargetCUDA();

  target_size_ = target_size;

  CUDA_CHECK(cudaMalloc(&d_target_, target_size_ * sizeof(float4)));
  CUDA_CHECK(cudaMemcpy(d_target_, target, target_size_ * sizeof(float4), cudaMemcpyHostToDevice));

  target_tree_ = cugicp_core_->compute_cutree(d_target_, target_size_);

  cugicp_core_->set_regularization_method(RegularizationMethod((int)regularization_method_));
  cugicp_core_->set_neighbors_distance(neighbors_distance_);

  d_target_covs_ = cugicp_core_->compute_covariances(d_target_, target_size_, target_tree_);
}

// ============================================================================
// Superclass function that work as adapters ONLY for compatibility to fastgicp
// ============================================================================

template <typename PointSource, typename PointTarget>
void FastCUGICP<PointSource, PointTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  if(cloud == input_) return; // Nothing to do
  
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud); // Fill also superclass attributes for compatibility

  if(sizeof(PointSource) != sizeof(float4))
  {
    std::cout << "CuGICP is incompatible with point type different from xyz_!";
    return;
  }

  setInputSource((float4*) cloud->points.data(), cloud->size()); 
}


template <typename PointSource, typename PointTarget>
void FastCUGICP<PointSource, PointTarget>::setSourceCovariances(
  const std::shared_ptr<std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>> covs)
{
  const size_t num_covs = covs->size();
  const size_t total_floats = num_covs * 16;
  const size_t total_bytes = total_floats * sizeof(float);

  // Free previous GPU memory (only if needed)
  if (d_source_covs_ != nullptr) {
    cudaFree(d_source_covs_);
    d_source_covs_ = nullptr;
  }

  // Allocate device memory
  CUDA_CHECK(cudaMalloc(&d_source_covs_, total_bytes));
  float* h_covariances_pinned = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_covariances_pinned, total_bytes));  // pinned memory


  // Fill pinned memory directly from Eigen matrices
  float* dst_ptr = h_covariances_pinned;
  for (size_t i = 0; i < num_covs; ++i) {
    const auto& m = (*covs)[i];
    // Write matrix in row-major float directly (no Eigen conversions)
    for (int row = 0; row < 4; ++row) {
      for (int col = 0; col < 4; ++col) {
        *dst_ptr++ = static_cast<float>(m(row, col));  // write row-by-row
      }
    }
  }

  // Copy pinned buffer to device
  CUDA_CHECK(cudaMemcpy(d_source_covs_, h_covariances_pinned, total_bytes, cudaMemcpyHostToDevice));

  // Free pinned memory
  CUDA_CHECK(cudaFreeHost(h_covariances_pinned));
}

template <typename PointSource, typename PointTarget>
void FastCUGICP<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  if(cloud == target_) return; // Nothing to do
  
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud); // Fill also superclass attributes for compatibility
  
  if(sizeof(PointTarget) != sizeof(float4))
  {
    std::cout << "CuGICP is incompatible with point types with size > 4!";
    return;
  }

  setInputTarget((float4*) cloud->points.data(), cloud->size()); 
}

template <typename PointSource, typename PointTarget>
void FastCUGICP<PointSource, PointTarget>::setTargetCovariances(const std::shared_ptr<std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>> covs) {
  const size_t num_covs = covs->size();
  const size_t total_floats = num_covs * 16;
  const size_t total_bytes = total_floats * sizeof(float);

  // Free previous GPU memory (only if needed)
  if (d_target_covs_ != nullptr) {
    cudaFree(d_target_covs_);
    d_target_covs_ = nullptr;
  }

  // Allocate device memory
  CUDA_CHECK(cudaMalloc(&d_target_covs_, total_bytes));
  float* h_covariances_pinned = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_covariances_pinned, total_bytes));  // pinned memory


  // Fill pinned memory directly from Eigen matrices
  float* dst_ptr = h_covariances_pinned;
  for (size_t i = 0; i < num_covs; ++i) {
    const auto& m = (*covs)[i];
    // Write matrix in row-major float directly (no Eigen conversions)
    for (int row = 0; row < 4; ++row) {
      for (int col = 0; col < 4; ++col) {
        *dst_ptr++ = static_cast<float>(m(row, col));  // write row-by-row
      }
    }
  }

  // Copy pinned buffer to device
  CUDA_CHECK(cudaMemcpy(d_target_covs_, h_covariances_pinned, total_bytes, cudaMemcpyHostToDevice));

  // Free pinned memory
  CUDA_CHECK(cudaFreeHost(h_covariances_pinned));
}

template <typename PointSource, typename PointTarget>
void FastCUGICP<PointSource, PointTarget>::swapSourceAndTarget() {
  FastGICP<PointSource, PointTarget>::swapSourceAndTarget();

  std::swap(d_source_, d_target_);
  std::swap(d_source_covs_, d_target_covs_);
  std::swap(source_size_, target_size_);
  std::swap(source_tree_, target_tree_);
}


template <typename PointSource, typename PointTarget>
void FastCUGICP<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {

  // Set params before iteration (TODO: Find a better method?)
  cugicp_core_->set_correspondence_randomness(k_correspondences_);
  cugicp_core_->set_max_correspondence_distance(corr_dist_threshold_);
  cugicp_core_->set_maximum_iterations(max_iterations_);
  cugicp_core_->set_transformation_epsilon(transformation_epsilon_);
  cugicp_core_->set_rotation_epsilon(rotation_epsilon_);

  // call align_cuda
  const Eigen::Matrix<float, 4, 4, Eigen::RowMajor> guess_rowmajor = guess;

  cugicp_core_->align_cuda(
    d_source_, d_source_covs_, source_size_,
    target_tree_, d_target_covs_,
    (float*) &guess_rowmajor);

  // Reconvert to column major
  float final_transformation_float[16];
  std::memcpy(final_transformation_float, cugicp_core_->get_final_transformation(), sizeof(float) * 16);
  final_transformation_ = Eigen::Map<Eigen::Matrix4f>(final_transformation_float).transpose(); // To colMajor

  converged_ = cugicp_core_->has_converged();
  nr_iterations_ = cugicp_core_->get_num_iterations();

  pcl::transformPointCloud(*input_, output, final_transformation_);
}


/** 
 * GETTERS
 * These functions copy data from GPU to CPU, so they can be slow.
 * Use them only for debugging or visualization purposes.
 */
// template <typename PointSource, typename PointTarget>
// const std::shared_ptr<std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>> 
//   FastCUGICP<PointSource, PointTarget>::getSourceCovariances() const {
    
//     if (!d_source_covs_ || source_size_ == 0) return nullptr;

//     // Allocate memory on CPU
//     auto source_covariances = std::make_shared<std::vector<Eigen::Matrix4d,
//                                                Eigen::aligned_allocator<Eigen::Matrix4d>>>(source_size_);

//     // Copy data from GPU to CPU
//     std::vector<float> h_source_covs(source_size_ * 16); // 4x4 row-major matrices
//     cudaMemcpy(h_source_covs.data(), d_source_covs_, source_size_ * 16 * sizeof(float), cudaMemcpyDeviceToHost);

//     // Efficient Eigen::Map-based conversion (Row-Major to Column-Major)
//     for (int i = 0; i < source_size_; ++i) {
//         Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> mapped_mat(&h_source_covs[i * 16]);
//         (*source_covariances)[i] = mapped_mat.transpose().cast<double>(); // Convert float → double
//     }

//     return source_covariances;
// }

// template <typename PointSource, typename PointTarget>
// const std::shared_ptr<std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>>
//   FastCUGICP<PointSource, PointTarget>::getTargetCovariances() const {

//     if (!d_target_covs_ || target_size_ == 0) return nullptr;

//     // Allocate memory on CPU
//     auto target_covariances = std::make_shared<std::vector<Eigen::Matrix4d,
//                                                Eigen::aligned_allocator<Eigen::Matrix4d>>>(target_size_);

//     // Copy data from GPU to CPU
//     std::vector<float> h_target_covs(target_size_ * 16); // 4x4 row-major matrices
//     cudaMemcpy(h_target_covs.data(), d_target_covs_, target_size_ * 16 * sizeof(float), cudaMemcpyDeviceToHost);

//     // Efficient Eigen::Map-based conversion (Row-Major to Column-Major)
//     for (int i = 0; i < target_size_; ++i) {
//         Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> mapped_mat(&h_target_covs[i * 16]);
//         (*target_covariances)[i] = mapped_mat.transpose().cast<double>(); // Convert float → double
//     }

//     return target_covariances;
// }

// template <typename PointSource, typename PointTarget>
// const std::shared_ptr<std::vector<Eigen::Vector3d>> 
// FastCUGICP<PointSource, PointTarget>::getSourceNormals() const {
//   const auto normals_float = getSourceNormalsf();
//   if (!normals_float) return nullptr;


//   std::shared_ptr<std::vector<Eigen::Vector3d>> source_normals = 
//     std::make_shared<std::vector<Eigen::Vector3d>>(normals_float->size());
//   for(size_t i = 0; i < normals_float->size(); i++)
//     source_normals->at(i) = normals_float->at(i).template cast<double>();;
//   return source_normals;
// }

// template <typename PointSource, typename PointTarget>
// const std::shared_ptr<std::vector<Eigen::Vector3f>> 
// FastCUGICP<PointSource, PointTarget>::getSourceNormalsf() const {

//   if (!d_source_covs_ || source_size_ == 0) return nullptr;

//   std::shared_ptr<std::vector<Eigen::Vector3f>> source_normals = 
//     std::make_shared<std::vector<Eigen::Vector3f>>(source_size_);

//   // Allocate device-side normal buffer (float3 equivalent)
//   float3* d_normals;
//   cudaMalloc(&d_normals, source_size_ * sizeof(float3));

//   // Launch the CUDA kernel (assumed to be handled inside cugicp_core_)
//   cugicp_core_->compute_normals_from_covariances(d_source_covs_, d_normals, source_size_);

//   // Allocate host-side float buffer
//   cudaMemcpy(source_normals->data(), d_normals, source_size_ * sizeof(float3), cudaMemcpyDeviceToHost);

//   // Cleanup
//   cudaFree(d_normals);

//   return source_normals;
// }

// template <typename PointSource, typename PointTarget>
// const std::shared_ptr<std::vector<Eigen::Vector3d>> 
// FastCUGICP<PointSource, PointTarget>::getTargetNormals() const {
//   const auto normals_float = getTargetNormalsf();
//   if (!normals_float) return nullptr;

//   std::shared_ptr<std::vector<Eigen::Vector3d>> target_normals = 
//     std::make_shared<std::vector<Eigen::Vector3d>>(normals_float->size());
//   for(size_t i = 0; i < normals_float->size(); i++)
//     target_normals->at(i) = normals_float->at(i).template cast<double>();;
//   return target_normals;
// }

// template <typename PointSource, typename PointTarget>
// const std::shared_ptr<std::vector<Eigen::Vector3f>> 
// FastCUGICP<PointSource, PointTarget>::getTargetNormalsf() const {

//   if (!d_target_covs_ || target_size_ == 0) return nullptr;

//   std::shared_ptr<std::vector<Eigen::Vector3f>> target_normals = 
//     std::make_shared<std::vector<Eigen::Vector3f>>(target_size_);

//   // Allocate device-side normal buffer
//   float3* d_normals;
//   cudaMalloc(&d_normals, target_size_ * sizeof(float3));

//   // Launch the CUDA kernel (assumed to be handled inside cugicp_core_)
//   cugicp_core_->compute_normals_from_covariances(d_target_covs_, d_normals, target_size_);

//   // Allocate host-side float buffer
//   cudaMemcpy(target_normals->data(), d_normals, target_size_ * sizeof(float3), cudaMemcpyDeviceToHost);

//   // Cleanup
//   cudaFree(d_normals);

//   return target_normals;
// }

}  // namespace fast_gicp

#endif
