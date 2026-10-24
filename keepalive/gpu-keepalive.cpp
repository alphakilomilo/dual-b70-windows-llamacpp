// gpu-keepalive: stop Windows from powering down idle Intel GPUs (runtime D3) and evicting
// llama-server's VRAM. On this dual-B70 box the headless card drops to D3 ~30-40 s after the
// last request; its 28 GB of weights/KV get evicted and the next request stalls ~17-19 s while
// they page back in (see llama.cpp #25646). A tiny device-side fill every few seconds keeps every
// GPU in D0. Targets ALL Level Zero GPUs, not just the default one (the display card never sleeps).
//
// Build: icx -fsycl -O2 gpu-keepalive.cpp -o gpu-keepalive.exe   (inside setvars)
// Usage: gpu-keepalive.exe [interval_seconds=10] [watch_process.exe]
// With a watch process the keepalive exits once that process is gone (or never appears within
// 5 minutes), so it cannot hold the GPUs awake after the server stops.
#include <sycl/sycl.hpp>
#define NOMINMAX
#include <windows.h>
#include <tlhelp32.h>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static bool process_running(const char * name) {
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) {
        return true; // cannot tell -- err on the side of staying alive
    }
    PROCESSENTRY32 pe{};
    pe.dwSize  = sizeof(pe);
    bool found = false;
    for (BOOL ok = Process32First(snap, &pe); ok && !found; ok = Process32Next(snap, &pe)) {
        found = _stricmp(pe.szExeFile, name) == 0;
    }
    CloseHandle(snap);
    return found;
}

int main(int argc, char ** argv) {
    const int    interval = argc > 1 ? std::max(1, std::atoi(argv[1])) : 10;
    const char * watch    = argc > 2 ? argv[2] : nullptr;

    std::vector<sycl::queue> queues;
    std::vector<int *>       bufs;
    for (const auto & dev : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        queues.emplace_back(dev);
        bufs.push_back(sycl::malloc_device<int>(64, queues.back()));
        std::printf("keepalive: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    }
    if (queues.empty()) {
        std::fprintf(stderr, "keepalive: no GPU devices found\n");
        return 1;
    }
    std::printf("keepalive: touching %zu GPU(s) every %d s\n", queues.size(), interval);
    std::fflush(stdout);

    const auto start = std::chrono::steady_clock::now();
    bool       seen  = false;
    for (unsigned tick = 0;; ++tick) {
        if (watch) {
            const bool running = process_running(watch);
            seen = seen || running;
            if (seen && !running) {
                std::printf("keepalive: %s exited, stopping\n", watch);
                break;
            }
            if (!seen && std::chrono::steady_clock::now() - start > std::chrono::minutes(5)) {
                std::printf("keepalive: %s never started, stopping\n", watch);
                break;
            }
        }
        for (size_t i = 0; i < queues.size(); ++i) {
            queues[i].fill(bufs[i], (int) tick, 64).wait();
        }
        std::this_thread::sleep_for(std::chrono::seconds(interval));
    }

    for (size_t i = 0; i < queues.size(); ++i) {
        sycl::free(bufs[i], queues[i]);
    }
    return 0;
}
