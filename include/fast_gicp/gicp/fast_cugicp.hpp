#ifndef FAST_GICP_FAST_CUGICP_HPP
#define FAST_GICP_FAST_CUGICP_HPP

#include <unordered_map>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

#include "fast_gicp/gicp/gicp_settings.hpp"
#include "fast_gicp/gicp/fast_gicp.hpp"
#include "cuda_runtime.h"

namespace fast_gicp {

class CuTree; // Forward Declaration: Declared in cugicp_core.cu

class CUGICP; // Forward Declaration: Declared in cugicp_core.cuh

//class CuTree; // All the CuTree pointers in setters have been replaced by char*, to avoid cuda dependencies

/**
 * @brief Fast Planar GICP algorithm boosted with OpenMP
 */
template<typename PointSource, typename PointTarget>
class FastCUGICP : public FastGICP<PointSource, PointTarget> {
public:
  using Scalar = float;
  using Matrix4 = typename pcl::Registration<PointSource, PointTarget, Scalar>::Matrix4;

  using PointCloudSource = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudSource;
  using PointCloudSourcePtr = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

  using PointCloudTarget = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudTarget;
  using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

  using pcl::Registration<PointSource, PointTarget, Scalar>::input_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::target_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::max_iterations_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::transformation_epsilon_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::nr_iterations_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::corr_dist_threshold_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::final_transformation_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::converged_;

  using LsqRegistration<PointSource, PointTarget>::rotation_epsilon_;
  using FastGICP<PointSource, PointTarget>::k_correspondences_;
  using FastGICP<PointSource, PointTarget>::regularization_method_;

  using pcl::Registration<PointSource, PointTarget, Scalar>::getFitnessScore;

public:

  FastCUGICP();
  virtual ~FastCUGICP() override;

  // Superclass function that work as adapters to the functions above
  virtual void setInputSource(const PointCloudSourceConstPtr& cloud) override;
  virtual void setSourceCovariances(const std::shared_ptr<std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>> covs);
  
  virtual void setInputTarget(const PointCloudTargetConstPtr& cloud) override;
  virtual void setTargetCovariances(const std::shared_ptr<std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>> covs);

  virtual void swapSourceAndTarget() override;
  virtual void clearSource() override;
  virtual void clearTarget() override;

  // virtual const std::shared_ptr<std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>> getSourceCovariances() const override;
  // virtual const std::shared_ptr<std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>> getTargetCovariances() const override;
  // virtual const std::shared_ptr<std::vector<Eigen::Vector3d>> getSourceNormals() const override;
  // virtual const std::shared_ptr<std::vector<Eigen::Vector3d>> getTargetNormals() const override;
  // virtual const std::shared_ptr<std::vector<Eigen::Vector3f>> getSourceNormalsf() const;
  // virtual const std::shared_ptr<std::vector<Eigen::Vector3f>> getTargetNormalsf() const;
  // virtual const std::shared_ptr<std::vector<uint8_t>> getSourceTypes() const;
  // virtual const std::shared_ptr<std::vector<uint8_t>> getTargetTypes() const; 

  // Obsolete functions
  virtual void setNumThreads(int /*n*/) {throw std::runtime_error("setNumThreads() NOT IMPLEMENTED IN CuGICP");}


protected:
  virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;

  template<typename PointT>
  bool calculate_covariances(const typename pcl::PointCloud<PointT>::ConstPtr& cloud, pcl::search::Search<PointT>& kdtree, std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances);
  
  virtual void setInputSource(
    float4* source,
    int source_size);

  virtual void setInputTarget(
    float4* target,
    int target_size);

  virtual void clearSourceCUDA();
  virtual void clearTargetCUDA();

  // ===================
  //   Persistent State
  // ===================
  std::unique_ptr<CUGICP> cugicp_core_ = nullptr; // Declared in cugicp_core.cuh

  // Source
  std::shared_ptr<CuTree> source_tree_ = nullptr;        ///< KD-tree for source cloud.
  float4* d_source_ = nullptr;           ///< GPU pointer to source cloud.
  float* d_source_covs_ = nullptr;       ///< GPU pointer to source covariances.
  int source_size_ = 0;

  // Target
  std::shared_ptr<CuTree> target_tree_ = nullptr;        ///< KD-tree for target cloud.
  float4* d_target_ = nullptr;           ///< GPU pointer to target cloud.
  float* d_target_covs_ = nullptr;       ///< GPU pointer to target covariances.
  int target_size_ = 0;

  // GICP Params
  float neighbors_distance_ = std::numeric_limits<float>::max();
};

} // namespace fast_gicp

#endif
