#include <drm_v3d.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>

#include "cblas-qmkl6.h"
#include "qmkl6-internal.hpp"

qmkl6_context qmkl6;

qmkl6_context::qmkl6_context(void) {
  int ret;

  ret = rpimemmgr_init(&rpimemmgr);
  if (ret) XERBLA(ret);

  unif_size = sizeof(uint32_t) * 1024;
  unif = (uint32_t *)alloc_memory(unif_size, unif_handle, unif_bus);

  drm_fd = rpimemmgr_borrow_drm_fd(&rpimemmgr);

  init_support();
  init_blas1();
  init_blas2();
  init_blas3();
  init_blaslike();
  init_fft();
}

qmkl6_context::~qmkl6_context(void) {
  int ret;

  mkl_set_exit_handler(exit);

  finalize_fft();
  finalize_blaslike();
  finalize_blas3();
  finalize_blas2();
  finalize_blas1();
  finalize_support();

  free_memory(unif);

  ret = rpimemmgr_finalize(&rpimemmgr);
  if (ret) XERBLA(ret);
}

void *qmkl6_context::alloc_memory(const size_t size, uint32_t &bus_addr) {
  void *virt_addr;

  const int ret = rpimemmgr_alloc_drm(size, &virt_addr, &bus_addr, &rpimemmgr);
  if (ret) XERBLA(ret);

  return virt_addr;
}

void *qmkl6_context::alloc_memory(const size_t size, uint32_t &handle,
                                  uint32_t &bus_addr) {
  void *const virt_addr = alloc_memory(size, bus_addr);

  handle = rpimemmgr_usraddr_to_handle(virt_addr, &rpimemmgr);

  return virt_addr;
}

void qmkl6_context::free_memory(void *const virt_addr) {
  const int ret = rpimemmgr_free_by_usraddr(virt_addr, &rpimemmgr);

  if (ret) XERBLA(ret);
}

uint32_t qmkl6_context::locate_virt(const void *const virt_addr) {
  const uint32_t bus_addr = rpimemmgr_usraddr_to_busaddr(virt_addr, &rpimemmgr);
  if (!bus_addr) XERBLA(1);

  return bus_addr;
}

uint32_t qmkl6_context::locate_virt(const void *const virt_addr,
                                    uint32_t &handle) {
  handle = rpimemmgr_usraddr_to_handle(virt_addr, &rpimemmgr);
  if (!handle) XERBLA(1);

  return locate_virt(virt_addr);
}

void qmkl6_context::execute_qpu_code(const uint32_t qpu_code_bus,
                                     const uint32_t unif_bus,
                                     const unsigned num_qpus,
                                     const unsigned num_handles, ...) {
  const uint32_t cfg[7] =
      {
          DRM_V3D_SET_FIELD(16, CSD_CFG0_NUM_WGS_X) |
              DRM_V3D_SET_FIELD(0, CSD_CFG0_WG_X_OFFSET),
          DRM_V3D_SET_FIELD(1, CSD_CFG1_NUM_WGS_Y) |
              DRM_V3D_SET_FIELD(0, CSD_CFG1_WG_Y_OFFSET),
          DRM_V3D_SET_FIELD(1, CSD_CFG2_NUM_WGS_Z) |
              DRM_V3D_SET_FIELD(0, CSD_CFG2_WG_Z_OFFSET),
          DRM_V3D_SET_FIELD(0, CSD_CFG3_MAX_SG_ID) |
              DRM_V3D_SET_FIELD(16 - 1, CSD_CFG3_BATCHES_PER_SG_M1) |
              DRM_V3D_SET_FIELD(16, CSD_CFG3_WGS_PER_SG) |
              DRM_V3D_SET_FIELD(16, CSD_CFG3_WG_SIZE),
          /* Number of batches, minus 1 */
          num_qpus - 1,
          /* Shader address, pnan, singleseg, threading, like a shader record.
           */
          qpu_code_bus,
          /* Uniforms address (4 byte aligned) */
          unif_bus,
      },
                 coef[4] = {0, 0, 0, 0};
  uint32_t handles[num_handles];

  va_list ap;
  va_start(ap, num_handles);
  for (unsigned i = 0; i < num_handles; ++i) handles[i] = va_arg(ap, uint32_t);
  va_end(ap);

  const int ret =
      drm_v3d_submit_csd(drm_fd, cfg, coef, handles, num_handles, 0, 0);
  if (ret) {
    fprintf(stderr, "error: drm_v3d_submit_csd: %s\n", strerror(errno));
    XERBLA(ret);
  }
}

void qmkl6_context::wait_for_handles(const uint64_t timeout_ns,
                                     const unsigned num_handles, ...) {
  va_list ap;
  va_start(ap, num_handles);
  for (unsigned i = 0; i < num_handles; ++i) {
    const uint32_t handle = va_arg(ap, uint32_t);
    const int ret = drm_v3d_wait_bo(drm_fd, handle, timeout_ns);
    if (ret) {
      fprintf(stderr, "error: drm_v3d_wait_bo: %s\n", strerror(errno));
      XERBLA(ret);
    }
  }
  va_end(ap);
}
