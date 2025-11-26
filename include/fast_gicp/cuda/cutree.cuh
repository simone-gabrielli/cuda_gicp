#ifndef FAST_GICP_CUTREE_CUH
#define FAST_GICP_CUTREE_CUH

#include "cukd/builder.h"
#include "cukd/spatial-kdtree.h"
#include "cukd/label_mask.h"
#include "cukd/traversal_label_pruned.h"

namespace cukd {
namespace labels {
// ---------------- data & traits ----------------
struct LPoint3f {
  float x, y, z, w;
  int label;
  int orig; // original index in the input array (for mapping to covariance indices)
};

struct LPoint3f_traits : public cukd::default_data_traits<float3> {
  using point_t = float3;
  static constexpr int num_dims = 3;
  enum { has_explicit_dim = false };
  __host__ __device__ static inline const float3 get_point(const LPoint3f& d) { return make_float3(d.x, d.y, d.z); }
  __host__ __device__ static inline float3 get_point(LPoint3f& d) { return make_float3(d.x, d.y, d.z); }
  __host__ __device__ static inline float get_coord(const LPoint3f& d, int dim) { return (dim == 0) ? d.x : (dim == 1) ? d.y : d.z; }
};


template <>
struct default_label_traits<LPoint3f> {
  __host__ __device__ static inline int get_label(const LPoint3f& d) { return d.label; }
};
}  // namespace labels
}  // namespace cukd


namespace fast_gicp {

// ======================================================
//                        SUPPORT
// ======================================================
using float4_t = cukd::default_data_traits<float4>;

class CuTree {
public:
    cukd::SpatialKDTree<float4, float4_t> tree;  // The actual KD-tree
};


class CuTreeLbl {
public:
    cukd::SpatialKDTree<cukd::labels::LPoint3f, cukd::labels::LPoint3f_traits> tree;  // The actual KD-tree
    cukd::labels::Mask* mask;
};

} // namespace fast_gicp

#endif // FAST_GICP_CUTREE_CUH