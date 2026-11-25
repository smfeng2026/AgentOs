/*
 * PocketEngine - thin, platform neutral wrapper around MNN's LLM runtime.
 *
 * The class owns one MNN::Transformer::Llm instance and keeps the chat
 * history, so multiple turns share the KV cache (prompt cache reuse).
 * Every turn reports token accounting and timing taken straight from
 * MNN's own LlmContext, so the numbers shown in the UI are the real
 * engine measurements, not wall clock guesses.
 */

#ifndef POCKET_LLM_ENGINE_H
#define POCKET_LLM_ENGINE_H

#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "llm/llm.hpp"

namespace pocket {

/** Token / latency accounting for a single turn. */
struct ChatStats {
    int promptTokens = 0;     // tokens prefilled for this turn (history-aware)
    int outputTokens = 0;     // tokens generated for this turn
    int contextTokens = 0;    // total tokens currently held in the KV cache
    long long prefillMs = 0;  // prompt processing time
    long long decodeMs = 0;   // token generation time
    long long ttfaMs = 0;     // time to first token
    long long totalMs = 0;    // wall clock of the whole turn (incl. sampling)
    long long loadMs = 0;     // model load time, reported for reference
    double prefillTps = 0.0;  // prompt tokens / second
    double decodeTps = 0.0;   // generated tokens / second

    std::string toJson() const;
};

class ChatEngine {
public:
    using DeltaFn = std::function<void(const std::string&)>;

    ChatEngine();
    ~ChatEngine();

    ChatEngine(const ChatEngine&) = delete;
    ChatEngine& operator=(const ChatEngine&) = delete;

    /**
     * Creates and loads the model described by <configPath>.
     * @param tmpDir      writable directory MNN may use for weight mmap cache
     * @param maxNewTokens default generation budget per turn
     */
    bool load(const std::string& configPath, const std::string& tmpDir,
              int maxNewTokens, std::string* error);

    /** Blocks until generation finishes, streaming deltas through onDelta. */
    bool chat(const std::string& prompt, const DeltaFn& onDelta,
              ChatStats* stats, std::string* error);

    void reset();
    void setSystemPrompt(const std::string& prompt);
    const std::string& systemPrompt() const { return systemPrompt_; }
    std::string describe() const;

    bool loaded() const { return llm_ != nullptr; }
    const std::string& lastError() const { return lastError_; }
    long long loadMs() const { return loadMs_; }

private:
    MNN::Transformer::Llm* llm_ = nullptr;
    MNN::Transformer::ChatMessages messages_;
    std::string systemPrompt_;
    std::string configPath_;
    std::string lastError_;
    long long loadMs_ = 0;
    int maxNewTokens_ = 512;
    std::mutex mutex_;
};

/** Escapes a UTF-8 string so it can be embedded in a JSON string literal. */
std::string jsonEscape(const std::string& input);

/** Cheap monotonic clock in microseconds. */
long long nowMicros();

/** Trims leading/trailing ASCII whitespace. */
std::string trim(const std::string& input);

} // namespace pocket

#endif // POCKET_LLM_ENGINE_H
