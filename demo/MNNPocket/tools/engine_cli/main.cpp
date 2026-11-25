/*
 * engine_cli - development harness for the desktop, used to validate the exact
 * same engine code that libllmbridge.so compiles for HarmonyOS.
 *
 * Chat mode (legacy):
 *   ./engine_cli <model_dir>/config.json <tmp_dir> "prompt" [prompt...]
 *
 * Decision modes:
 *   ./engine_cli decide <model_dir> <request.json> [threads] [max_tokens]
 *       Runs the single-forward decision graph, prints the result JSON and
 *       the timing report the app renders under each result card.
 *   ./engine_cli encode <model_dir> <request.json>
 *       Encoding-only dump (ids / positions / segments / pointer positions /
 *       mask hash) for golden-vector parity against inference_mnn.py.
 */

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "decision_engine.h"
#include "llm_engine.h"

namespace {

bool readWholeFile(const std::string& path, std::string* out) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        return false;
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    *out = ss.str();
    return true;
}

int runDecision(int argc, char** argv, bool dumpOnly) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s %s <model_dir> <request.json> [threads] [max_tokens]\n",
                     argv[0], dumpOnly ? "encode" : "decide");
        return 2;
    }
    const std::string modelDir = argv[2];
    std::string request;
    if (!readWholeFile(argv[3], &request)) {
        std::fprintf(stderr, "cannot read request: %s\n", argv[3]);
        return 2;
    }
    const int threads = argc >= 5 ? std::atoi(argv[4]) : 4;
    const int maxTokens = argc >= 6 ? std::atoi(argv[5]) : 32768;

    pocket::DecisionEngine engine;
    std::string error;
    if (!engine.load(modelDir, "decision-w8-q8-embedding.mnn", threads, maxTokens, &error)) {
        std::fprintf(stderr, "[load] failed: %s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr, "[load] ok in %lld ms\n", engine.loadMs());

    if (dumpOnly) {
        std::string dump;
        if (!engine.encodeDump(request, &dump, &error)) {
            std::fprintf(stderr, "[encode] failed: %s\n", error.c_str());
            return 1;
        }
        std::printf("%s\n", dump.c_str());
        return 0;
    }

    std::string result;
    pocket::DecisionStats stats;
    if (!engine.predict(request, &result, &stats, &error)) {
        std::fprintf(stderr, "[decide] failed: %s\n", error.c_str());
        return 1;
    }
    std::printf("%s\n", result.c_str());
    std::fprintf(stderr,
                 "[stat] input=%d tok questions=%d encode=%lld ms forward=%lld ms T=%.6f\n",
                 stats.inputTokens, stats.questionCount, stats.encodeMs, stats.forwardMs,
                 stats.temperature);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "decide") == 0) {
        return runDecision(argc, argv, false);
    }
    if (argc >= 2 && std::strcmp(argv[1], "encode") == 0) {
        return runDecision(argc, argv, true);
    }
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s <config.json> <tmp_dir> <prompt> [prompt...]\n"
                     "       %s decide <model_dir> <request.json> [threads] [max_tokens]\n"
                     "       %s encode <model_dir> <request.json>\n",
                     argv[0], argv[0], argv[0]);
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
