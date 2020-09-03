#include <cmath>
#include <cstdio>

#include "cblas-qmkl6.h"
#include "qmkl6-internal.hpp"


static const uint64_t qpu_saxpy_orig[] = {
#include "saxpy.qhex6"
};

static const uint64_t qpu_scopy_orig[] = {
#include "scopy.qhex6"
};

static const uint64_t qpu_sdot_orig[] = {
#include "sdot.qhex6"
};

static const uint64_t qpu_snrm2_orig[] = {
#include "snrm2.qhex6"
};


void cblas_saxpy(int n, const float a, const float *x, const int incx, float *y,
        const int incy)
{
    if (n <= 0) {
        fprintf(stderr, "error: n (%d) must be greater than zero\n", n);
        XERBLA(1);
    }
    if (incx <= 0 || incy <= 0) {
        fprintf(stderr, "error: inc must be greater than zero for now\n");
        XERBLA(1);
    }

    const unsigned num_queues = 4, num_threads = 16, num_qpus = 8,
          unroll = 1 << 1, align = num_queues * num_threads * num_qpus * unroll;
    const int n_rem = n % align;
    n -= n_rem;

    uint32_t x_handle, y_handle, x_bus, y_bus;

    if (n > 0) {
        qmkl6.locate_virt((void*) x, x_handle, x_bus);
        qmkl6.locate_virt((void*) y, y_handle, y_bus);

        qmkl6.unif[0] = n;
        qmkl6.unif[1] = qmkl6.bit_cast <uint32_t> (a);
        qmkl6.unif[2] = x_bus;
        qmkl6.unif[3] = incx;
        qmkl6.unif[4] = y_bus;
        qmkl6.unif[5] = incy;

        qmkl6.execute_qpu_code(qmkl6.qpu_saxpy_bus, qmkl6.unif_bus, num_qpus, 1,
                y_handle);
    }

    for (int i = 0, j = incx * n, k = incy * n; i < n_rem;
            ++i, j += incx, k += incy)
        y[k] += a * x[j];

    if (n > 0)
        qmkl6.wait_for_handles(qmkl6.timeout_ns, 1, y_handle);
}

void cblas_scopy(int n, const float * const x, const int incx, float * const y,
        const int incy)
{
    if (n <= 0) {
        fprintf(stderr, "error: n (%d) must be greater than zero\n", n);
        XERBLA(1);
    }
    if (incx <= 0 || incy <= 0) {
        fprintf(stderr, "error: inc must be greater than zero for now\n");
        XERBLA(1);
    }

    const unsigned num_queues = 8, num_threads = 16, num_qpus = 8,
          unroll = 1 << 0, align = num_queues * num_threads * num_qpus * unroll;
    const int n_rem = n % align;
    n -= n_rem;

    uint32_t x_handle, y_handle, x_bus, y_bus;

    if (n > 0) {
        qmkl6.locate_virt((void*) x, x_handle, x_bus);
        qmkl6.locate_virt((void*) y, y_handle, y_bus);

        qmkl6.unif[0] = n;
        qmkl6.unif[1] = x_bus;
        qmkl6.unif[2] = incx;
        qmkl6.unif[3] = y_bus;
        qmkl6.unif[4] = incy;

        qmkl6.execute_qpu_code(qmkl6.qpu_scopy_bus, qmkl6.unif_bus, 8, 1,
                y_handle);
    }

    for (int i = 0, j = incx * n, k = incy * n; i < n_rem;
            ++i, j += incx, k += incy)
        y[k] = x[j];

    if (n > 0)
        qmkl6.wait_for_handles(qmkl6.timeout_ns, 1, y_handle);
}

float cblas_sdot(int n, const float *x, const int incx, const float *y,
        const int incy)
{
    if (n <= 0) {
        fprintf(stderr, "error: n (%d) must be greater than zero\n", n);
        XERBLA(1);
    }
    if (incx <= 0 || incy <= 0) {
        fprintf(stderr, "error: inc must be greater than zero for now\n");
        XERBLA(1);
    }

    const unsigned num_queues = 4, num_threads = 16, num_qpus = 8,
          unroll = 1 << 4, align = num_queues * num_threads * num_qpus * unroll;
    const int n_rem = n % align;
    n -= n_rem;

    uint32_t x_handle, y_handle, x_bus, y_bus;

    if (n > 0) {
        qmkl6.locate_virt((void*) x, x_handle, x_bus);
        qmkl6.locate_virt((void*) y, y_handle, y_bus);

        qmkl6.unif[0] = n;
        qmkl6.unif[1] = x_bus;
        qmkl6.unif[2] = incx;
        qmkl6.unif[3] = y_bus;
        qmkl6.unif[4] = incy;
        qmkl6.unif[5] = qmkl6.unif_bus;

        qmkl6.execute_qpu_code(qmkl6.qpu_sdot_bus, qmkl6.unif_bus, num_qpus, 1,
                qmkl6.unif_handle);
    }

    float result = 0.f;
    for (int i = 0, j = incx * n, k = incy * n; i < n_rem;
            ++i, j += incx, k += incy)
        result += x[j] * y[k];

    if (n > 0) {
        qmkl6.wait_for_handles(qmkl6.timeout_ns, 1, qmkl6.unif_handle);

        const float * const results = (float*) qmkl6.unif;
        for (unsigned i = 0; i < num_threads * num_qpus; ++i)
            result += results[i];
    }

    return result;
}

float cblas_snrm2(int n, const float *x, const int incx)
{
    if (n <= 0) {
        fprintf(stderr, "error: n (%d) must be greater than zero\n", n);
        XERBLA(1);
    }
    if (incx <= 0) {
        fprintf(stderr, "error: inc must be greater than zero for now\n");
        XERBLA(1);
    }

    const unsigned num_queues = 8, num_threads = 16, num_qpus = 8,
          unroll = 1 << 5, align = num_queues * num_threads * num_qpus * unroll;
    const int n_rem = n % align;
    n -= n_rem;

    uint32_t x_handle, x_bus;

    if (n > 0) {
        qmkl6.locate_virt((void*) x, x_handle, x_bus);

        qmkl6.unif[0] = n;
        qmkl6.unif[1] = x_bus;
        qmkl6.unif[2] = incx;
        qmkl6.unif[3] = qmkl6.unif_bus;

        qmkl6.execute_qpu_code(qmkl6.qpu_snrm2_bus, qmkl6.unif_bus, num_qpus, 1,
                qmkl6.unif_handle);
    }

    float result = 0.f;
    for (int i = 0, j = incx * n; i < n_rem; ++i, j += incx)
        result += x[j] * x[j];

    if (n > 0) {
        qmkl6.wait_for_handles(qmkl6.timeout_ns, 1, qmkl6.unif_handle);

        const float * const results = (float*) qmkl6.unif;
        for (unsigned i = 0; i < 16 * num_qpus; ++i)
            result += results[i];
    }

    return std::sqrt(result);
}

void qmkl6_context::init_blas1(void)
{
    qpu_saxpy = (uint64_t*) alloc_memory(sizeof(qpu_saxpy_orig),
            qpu_saxpy_handle, qpu_saxpy_bus);
    memcpy(qpu_saxpy, qpu_saxpy_orig, sizeof(qpu_saxpy_orig));

    qpu_scopy = (uint64_t*) alloc_memory(sizeof(qpu_scopy_orig),
            qpu_scopy_handle, qpu_scopy_bus);
    memcpy(qpu_scopy, qpu_scopy_orig, sizeof(qpu_scopy_orig));

    qpu_sdot = (uint64_t*) alloc_memory(sizeof(qpu_sdot_orig), qpu_sdot_handle,
            qpu_sdot_bus);
    memcpy(qpu_sdot, qpu_sdot_orig, sizeof(qpu_sdot_orig));

    qpu_snrm2 = (uint64_t*) alloc_memory(sizeof(qpu_snrm2_orig),
            qpu_snrm2_handle, qpu_snrm2_bus);
    memcpy(qpu_snrm2, qpu_snrm2_orig, sizeof(qpu_snrm2_orig));
}

void qmkl6_context::finalize_blas1(void)
{
    free_memory(sizeof(qpu_snrm2_orig), qpu_snrm2_handle, qpu_snrm2);
    free_memory(sizeof(qpu_sdot_orig), qpu_sdot_handle, qpu_sdot);
    free_memory(sizeof(qpu_scopy_orig), qpu_scopy_handle, qpu_scopy);
    free_memory(sizeof(qpu_saxpy_orig), qpu_saxpy_handle, qpu_saxpy);
}
