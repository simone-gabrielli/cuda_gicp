#include <fast_gicp/gicp/fast_cugicp.hpp>
#include <fast_gicp/gicp/impl/fast_cugicp_impl.hpp>

template class fast_gicp::FastCUGICP<pcl::PointXYZ, pcl::PointXYZ>;
// Compatible only with PointXYZ as of now
// template class fast_gicp::FastGICP<pcl::PointXYZI, pcl::PointXYZI>;