#ifndef CUGICP_CLASSIFIER_CUH
#define CUGICP_CLASSIFIER_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include <fast_gicp/gicp/gicp_settings.hpp>

__global__ void classify_curvature_kernel(
    const float4* __restrict__ d_points,
    uint8_t* d_types,
    int N,
    float edge_threshold,
    float plane_threshold,
    float max_dist)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 5 || idx >= N - 5) return; // skip boundary cases

    float4 pi = d_points[idx];
    float range_i = sqrtf(pi.x * pi.x + pi.y * pi.y + pi.z * pi.z);
    float sum_distances = 0.0f;

    for (int j = -5; j <= 5; ++j) {
        if (j == 0) continue;
        float4 pj = d_points[idx + j];
        float range_j = sqrtf(pj.x * pj.x + pj.y * pj.y + pj.z * pj.z);
        float dist = fabsf(range_i - range_j);
        sum_distances += (dist <= max_dist) ? dist : max_dist;
    }

    float curvature = sum_distances * sum_distances;

    if (curvature > edge_threshold) {
        d_types[idx] = 2; // Edge
    } else if (curvature < plane_threshold) {
        d_types[idx] = 1; // Plane
    } else {
        d_types[idx] = 0; // None
    }
}

// For cloud classification
__global__ void classify_normals_kernel(
    const float4* __restrict__ d_points,
    const float3* __restrict__ d_normals,
    uint8_t* d_types,
    int N,
    const float max_angle,
    const float3 range)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float4 p = d_points[idx];
    float3 n = d_normals[idx];

    float norm = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
    if (norm > 1.001f || norm < 0.999f) {
        d_types[idx] = static_cast<uint8_t>(fast_gicp::PointClass::INVALID);
        return;
    }

    if (fabsf(p.x) > range.x || fabsf(p.y) > range.y || p.z > range.z) {
        d_types[idx] = static_cast<uint8_t>(fast_gicp::PointClass::UNCLASSIFIED);
        return;
    }

    float angle = fabsf(atan2f(n.z, hypotf(n.x, n.y)));

    if (angle < max_angle) {
        d_types[idx] = static_cast<uint8_t>(fast_gicp::PointClass::WALL);
    } else if (fabsf(M_PI_2 - angle) < max_angle) {
        d_types[idx] = static_cast<uint8_t>(fast_gicp::PointClass::GROUND);
    } else {
        d_types[idx] = static_cast<uint8_t>(fast_gicp::PointClass::UNCLASSIFIED);
    }
}

// For sector classification
__global__ void classify_normals_kernel(
    const float4*    __restrict__ d_points,
    const float3*    __restrict__ d_normals,
    int                           N,
    const uint16_t*  __restrict__ d_idx_ref_line,
    const float*     __restrict__ d_track_x,
    const float*     __restrict__ d_track_y,
    const float*     __restrict__ d_track_z,
    const float*     __restrict__ d_track_roll,
    uint8_t*         __restrict__ d_types,
    const float       max_angle)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // 0) Default to UNCLASSIFIED (matches CPU preassign)
    d_types[idx] = static_cast<uint8_t>(fast_gicp::PointClass::UNCLASSIFIED);

    // 1) Load point & normal
    float4 p = d_points[idx];
    float3 n = d_normals[idx];

    // 2) Normal‐length check
    float norm = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
    if (norm > 1.001f || norm < 0.999f) {
        d_types[idx] = static_cast<uint8_t>(fast_gicp::PointClass::INVALID);
        return;
    }

    // 3) Grab precomputed ref‐line data
    uint16_t ref_i = d_idx_ref_line[idx];
    float x_ref   = d_track_x[ref_i];
    float y_ref   = d_track_y[ref_i];
    float z_ref   = d_track_z[ref_i];
    float roll    = d_track_roll[ref_i];

    // 4) Range filtering (circular + vertical)
    float dx = x_ref - p.x;
    float dy = y_ref - p.y;
    float dist_xy = sqrtf(dx*dx + dy*dy);
    float dist_z  = fabsf(p.z - z_ref);
    
    // 5) Angle‐with‐banking exactly as on the CPU
    float az = n.z;                     // normal’s z‐component
    float hor = fabsf(asinf(az));       // |asin(nz)|
    float hor_diff = fabsf(hor - fabsf(roll));
    if (hor_diff < max_angle) {
        d_types[idx] = static_cast<uint8_t>(fast_gicp::PointClass::WALL);
    } else {
        float ver = fabsf(acosf(az));   // |acos(nz)|
        float ver_diff = fabsf(ver - fabsf(roll));
        if (ver_diff < max_angle) {
            d_types[idx] = static_cast<uint8_t>(fast_gicp::PointClass::GROUND);
        }
    }
}



#endif // CUGICP_CLASSIFIER_CUH