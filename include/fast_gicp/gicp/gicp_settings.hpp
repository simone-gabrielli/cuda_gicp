#ifndef FAST_GICP_GICP_SETTINGS_HPP
#define FAST_GICP_GICP_SETTINGS_HPP

namespace fast_gicp {

enum class RegularizationMethod { NONE, MIN_EIG, NORMALIZED_MIN_EIG, PLANE, FROBENIUS };

enum class NeighborSearchMethod { DIRECT27, DIRECT7, DIRECT1, /* supported on only VGICP_CUDA */ DIRECT_RADIUS };

enum class VoxelAccumulationMode { ADDITIVE, ADDITIVE_WEIGHTED, MULTIPLICATIVE };

static constexpr float COV_33_INVALID = -999.f;

struct WGICPCudaOptions {
  float* d_weight_map = nullptr;
  uint8_t* d_source_types = nullptr;
  uint8_t* d_target_types = nullptr;
};

enum PointClass
{
  UNCLASSIFIED = 0,
  WALL = 1,
  GROUND = 2,
  INVALID = 15
};
}

#endif