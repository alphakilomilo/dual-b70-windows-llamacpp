// Does launching one kernel alternately on two GPUs leak host memory? (2026-10-23 leak hunt)
// usage: alt.exe <mode> <launches>   mode: same | alt | alt_sync (alternate + wait per switch)
#include <sycl/sycl.hpp>
#include <windows.h>
#include <psapi.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <thread>

static double private_mb() {
    PROCESS_MEMORY_COUNTERS_EX pmc = {};
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *) &pmc, sizeof(pmc));
    return pmc.PrivateUsage / 1048576.0;
}

class add_kernel_with_a_long_name_so_the_mangled_string_is_easy_to_spot;
template <int D> class add_kernel_per_device;

int main(int argc, char ** argv) {
    const char * mode = argc > 1 ? argv[1] : "alt";
    const long n = argc > 2 ? atol(argv[2]) : 100000;
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.size() < 2) { printf("need 2 GPUs, have %zu\n", devs.size()); return 1; }
    const bool sepctx = !strcmp(mode, "alt_sepctx");
    sycl::queue q0 = sepctx ? sycl::queue(sycl::context(devs[0]), devs[0], sycl::property::queue::in_order())
                            : sycl::queue(devs[0], sycl::property::queue::in_order());
    sycl::queue q1 = sepctx ? sycl::queue(sycl::context(devs[1]), devs[1], sycl::property::queue::in_order())
                            : sycl::queue(devs[1], sycl::property::queue::in_order());
    printf("same context: %d\n", (int) (q0.get_context() == q1.get_context()));
    float * a0 = sycl::malloc_device<float>(1024, q0);
    float * a1 = sycl::malloc_device<float>(1024, q1);
    auto launch = [](sycl::queue & q, float * a) {
        q.parallel_for<add_kernel_with_a_long_name_so_the_mangled_string_is_easy_to_spot>(sycl::range<1>(1024), [=](sycl::id<1> i) { a[i] += 1.0f; });
    };
    launch(q0, a0); launch(q1, a1); q0.wait(); q1.wait();
    const double m0 = private_mb();
    for (long i = 0; i < n; i++) {
        if (!strcmp(mode, "same")) {
            launch(q0, a0);
        } else if (!strcmp(mode, "alt_names")) {
            if (i & 1) {
                q1.parallel_for<add_kernel_per_device<1>>(sycl::range<1>(1024), [=](sycl::id<1> j) { a1[j] += 1.0f; });
            } else {
                q0.parallel_for<add_kernel_per_device<0>>(sycl::range<1>(1024), [=](sycl::id<1> j) { a0[j] += 1.0f; });
            }
        } else if (!strcmp(mode, "alt_threads")) {
            break;
        } else {
            launch((i & 1) ? q1 : q0, (i & 1) ? a1 : a0);
            if (!strcmp(mode, "alt_sync") && (i % 1000) == 999) { q0.wait(); q1.wait(); }
        }
        if ((i % 20000) == 19999) { q0.wait(); q1.wait(); }
    }
    if (!strcmp(mode, "alt_threads")) {
        // one host thread per device, launching in lockstep (both devices still interleave in time)
        std::thread t0([&] { for (long i = 0; i < n / 2; i++) { launch(q0, a0); if ((i % 10000) == 9999) q0.wait(); } });
        std::thread t1([&] { for (long i = 0; i < n / 2; i++) { launch(q1, a1); if ((i % 10000) == 9999) q1.wait(); } });
        t0.join(); t1.join();
    }
    q0.wait(); q1.wait();
    const double m1 = private_mb();
    printf("mode %s launches %ld: private %.1f -> %.1f MB, %.1f bytes per launch\n", mode, n, m0, m1, (m1 - m0) * 1048576.0 / n);
    return 0;
}
