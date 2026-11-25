/*
 * engine_cli - development harness for the desktop, used to validate the exact
 * same ChatEngine code that libllmbridge.so compiles for HarmonyOS.
 *
 * It loads a model directory, streams one answer to stdout and prints the
 * token/latency report that the app renders under each bubble.
 *
 *   ./engine_cli <model_dir>/config.json <tmp_dir> "prompt" [max_new_tokens]
 */

#include <cstdio>
#include <cstring>
#include <string>

#include "llm_engine.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s <config.json> <tmp_dir> <prompt> [prompt...]\n",
                     argv[0]);
        return 2;
    }
    const std::string configPath = argv[1];
    const std::string tmpDir = argv[2];

    pocket::ChatEngine engine;
    std::string error;
    std::printf("[load] %s\n", configPath.c_str());
    if (!engine.load(configPath, tmpDir, 128, &error)) {
        std::fprintf(stderr, "[load] failed: %s\n", error.c_str());
        return 1;
    }
    std::printf("[load] ok in %lld ms\n", engine.loadMs());

    for (int i = 3; i < argc; ++i) {
        const std::string prompt = argv[i];
        pocket::ChatStats stats;
        std::printf("[ask%d] %s\n[ans%d] ", i - 2, prompt.c_str(), i - 2);
        std::fflush(stdout);
        const bool ok = engine.chat(prompt, [](const std::string& text) {
            std::printf("%s", text.c_str());
            std::fflush(stdout);
        }, &stats, &error);
        std::printf("\n");
        if (!ok) {
            std::fprintf(stderr, "[chat] failed: %s\n", error.c_str());
            return 1;
        }
        std::printf("[stat%d] %s\n\n", i - 2, stats.toJson().c_str());
    }
    return 0;
}
