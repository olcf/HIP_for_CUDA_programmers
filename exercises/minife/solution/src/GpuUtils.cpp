#include <GpuUtils.h>

namespace miniFE {

  hipStream_t GpuManager::s1;
  hipStream_t GpuManager::s2;
  hipEvent_t GpuManager::e1;
  hipEvent_t GpuManager::e2;
  bool GpuManager::initialized=false;

}
