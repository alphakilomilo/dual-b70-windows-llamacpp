// Workaround test (2026-10-23): one SYCL context per GPU, both wrapping ONE Level Zero context.
// 1) leak per alternating launch  2) GPU1 kernel reads/writes host USM allocated through GPU0's SYCL context
// 3) GPU1 memcpy from that host USM  4) GPU0 device memory -> host (ctx0) -> GPU1 device memory round trip
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <windows.h>
#include <psapi.h>
#include <cstdio>
#include <vector>

static double private_mb() {
    PROCESS_MEMORY_COUNTERS_EX pmc = {};
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *) &pmc, sizeof(pmc));
    return pmc.PrivateUsage / 1048576.0;
}
class shared_ze_add_kernel;

int main(int argc, char ** argv) {
    const long n = argc > 1 ? atol(argv[1]) : 300000;
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.size() < 2) { printf("need 2 GPUs\n"); return 1; }
    namespace l0 = sycl::ext::oneapi::level_zero;
    // one native L0 context spanning both devices, via the platform default context
    sycl::context both(devs);
    ze_context_handle_t zctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(both);
    auto mk = [&](const sycl::device & d) {
        return sycl::make_context<sycl::backend::ext_oneapi_level_zero>(
            sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::context>{zctx, {d}, l0::ownership::keep});
    };
    sycl::context c0 = mk(devs[0]), c1 = mk(devs[1]);
    sycl::queue q0(c0, devs[0], sycl::property::queue::in_order());
    sycl::queue q1(c1, devs[1], sycl::property::queue::in_order());
    printf("contexts differ: %d, native shared: %d\n", (int) (c0 != c1),
           (int) (sycl::get_native<sycl::backend::ext_oneapi_level_zero>(c0) == sycl::get_native<sycl::backend::ext_oneapi_level_zero>(c1)));

    // 2) + 3) + 4): host buffer allocated through GPU0's context, used by GPU1
    const size_t N = 4096;
    float * h0 = sycl::malloc_host<float>(N, q0);
    float * d0 = sycl::malloc_device<float>(N, q0);
    float * d1 = sycl::malloc_device<float>(N, q1);
    for (size_t i = 0; i < N; i++) h0[i] = (float) i;
    q1.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) { h0[i] = h0[i] * 2.0f + 1.0f; }).wait();   // GPU1 kernel on ctx0 host USM
    bool ok_kernel = true;
    for (size_t i = 0; i < N; i++) ok_kernel &= h0[i] == (float) i * 2.0f + 1.0f;
    q0.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) { d0[i] = (float) i + 0.5f; }).wait();
    q0.memcpy(h0, d0, N * sizeof(float)).wait();                                                     // GPU0 -> host
    q1.memcpy(d1, h0, N * sizeof(float)).wait();                                                     // host(ctx0) -> GPU1
    std::vector<float> back(N);
    q1.memcpy(back.data(), d1, N * sizeof(float)).wait();
    bool ok_copy = true;
    for (size_t i = 0; i < N; i++) ok_copy &= back[i] == (float) i + 0.5f;
    printf("GPU1 kernel on ctx0 host USM: %s, GPU0->host->GPU1 copy: %s\n", ok_kernel ? "OK" : "FAIL", ok_copy ? "OK" : "FAIL");

    // 1) leak per alternating launch
    float * a0 = d0; float * a1 = d1;
    auto launch = [](sycl::queue & q, float * a) {
        q.parallel_for<shared_ze_add_kernel>(sycl::range<1>(1024), [=](sycl::id<1> i) { a[i] += 1.0f; });
    };
    launch(q0, a0); launch(q1, a1); q0.wait(); q1.wait();
    const double m0 = private_mb();
    for (long i = 0; i < n; i++) {
        launch((i & 1) ? q1 : q0, (i & 1) ? a1 : a0);
        if ((i % 20000) == 19999) { q0.wait(); q1.wait(); }
    }
    q0.wait(); q1.wait();
    const double m1 = private_mb();
    printf("alternating launches %ld: private %.1f -> %.1f MB, %.1f bytes per launch\n", n, m0, m1, (m1 - m0) * 1048576.0 / n);
    sycl::free(h0, q0); sycl::free(d0, q0); sycl::free(d1, q1);
    return 0;
}
