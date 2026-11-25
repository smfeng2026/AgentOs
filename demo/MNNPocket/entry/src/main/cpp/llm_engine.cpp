#include "llm_engine.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ostream>
#include <sstream>
#include <streambuf>

namespace pocket {

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

long long nowMicros() {
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

std::string jsonEscape(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 8);
    for (char c : input) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

std::string trim(const std::string& input) {
    const char* ws = " \t\r\n";
    size_t begin = input.find_first_not_of(ws);
    if (begin == std::string::npos) {
        return std::string();
    }
    size_t end = input.find_last_not_of(ws);
    return input.substr(begin, end - begin + 1);
}

std::string ChatStats::toJson() const {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "{\"promptTokens\":%d,\"outputTokens\":%d,\"contextTokens\":%d,"
                  "\"prefillMs\":%lld,\"decodeMs\":%lld,\"ttfaMs\":%lld,\"totalMs\":%lld,"
                  "\"loadMs\":%lld,\"prefillTps\":%.1f,\"decodeTps\":%.1f}",
                  promptTokens, outputTokens, contextTokens, prefillMs, decodeMs, ttfaMs,
                  totalMs, loadMs, prefillTps, decodeTps);
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// streaming plumbing
// ---------------------------------------------------------------------------

namespace {

/** std::streambuf that forwards every write to a callback. */
class StreamBuffer : public std::streambuf {
public:
    using Callback = std::function<void(const char*, size_t)>;

    explicit StreamBuffer(Callback callback) : callback_(std::move(callback)) {}

protected:
    std::streamsize xsputn(const char* data, std::streamsize size) override {
        if (callback_ && size > 0) {
            callback_(data, static_cast<size_t>(size));
        }
        return size;
    }

    int_type overflow(int_type ch) override {
        if (ch != traits_type::eof() && callback_) {
            char c = static_cast<char>(ch);
            callback_(&c, 1);
        }
        return ch;
    }

private:
    Callback callback_;
};

/**
 * Ships only complete UTF-8 sequences downstream. Byte-fallback tokenizers can
 * hand us a single byte of a 3-byte CJK character; emitting it immediately would
 * render a replacement glyph, so incomplete tails are buffered for the next call.
 */
class Utf8Emitter {
public:
    using Sink = std::function<void(const std::string&)>;

    explicit Utf8Emitter(Sink sink) : sink_(std::move(sink)) {}

    void push(const char* data, size_t size) {
        pending_.append(data, size);
        size_t complete = completePrefix(pending_);
        if (complete > 0) {
            std::string ready = pending_.substr(0, complete);
            pending_.erase(0, complete);
            if (sink_) {
                sink_(ready);
            }
        }
    }

    void flush() {
        if (!pending_.empty()) {
            std::string rest;
            rest.swap(pending_);
            if (sink_) {
                sink_(rest);
            }
        }
    }

private:
    static size_t completePrefix(const std::string& text) {
        size_t index = 0;
        size_t lastComplete = 0;
        const size_t total = text.size();
        while (index < total) {
            unsigned char c = static_cast<unsigned char>(text[index]);
            size_t need = 1;
            if (c < 0x80) {
                need = 1;
            } else if ((c >> 5) == 0x06) {
                need = 2;
            } else if ((c >> 4) == 0x0E) {
                need = 3;
            } else if ((c >> 3) == 0x1E) {
                need = 4;
            } else {
                // Stray continuation / invalid lead byte: pass it through.
                index += 1;
                lastComplete = index;
                continue;
            }
            if (index + need > total) {
                break;
            }
            index += need;
            lastComplete = index;
        }
        return lastComplete;
    }

    Sink sink_;
    std::string pending_;
};

constexpr int kMaxHistoryMessages = 13; // system + 6 rounds

} // namespace

// ---------------------------------------------------------------------------
// ChatEngine
// ---------------------------------------------------------------------------

ChatEngine::ChatEngine() = default;

ChatEngine::~ChatEngine() {
    if (llm_ != nullptr) {
        delete llm_;
        llm_ = nullptr;
    }
}

void ChatEngine::setSystemPrompt(const std::string& prompt) {
    std::lock_guard<std::mutex> lock(mutex_);
    systemPrompt_ = prompt;
    messages_.clear();
    if (!systemPrompt_.empty()) {
        messages_.emplace_back("system", systemPrompt_);
    }
}

bool ChatEngine::load(const std::string& configPath, const std::string& tmpDir,
                      int maxNewTokens, std::string* error) {
    std::lock_guard<std::mutex> lock(mutex_);
    lastError_.clear();
    configPath_ = configPath;
    maxNewTokens_ = maxNewTokens > 0 ? maxNewTokens : 512;

    const long long begin = nowMicros();
    llm_ = MNN::Transformer::Llm::createLLM(configPath);
    if (llm_ == nullptr) {
        lastError_ = "无法读取模型配置: " + configPath;
        if (error != nullptr) {
            *error = lastError_;
        }
        return false;
    }

    // tmp_path must be inside the app sandbox: MNN mmaps weight cache files there.
    std::ostringstream cfg;
    cfg << "{\"tmp_path\":\"" << jsonEscape(tmpDir) << "\",\"max_new_tokens\":" << maxNewTokens_ << "}";
    llm_->set_config(cfg.str());

    if (!llm_->load()) {
        lastError_ = "模型加载失败（内存不足或文件损坏）";
        delete llm_;
        llm_ = nullptr;
        if (error != nullptr) {
            *error = lastError_;
        }
        return false;
    }

    loadMs_ = (nowMicros() - begin) / 1000;
    if (systemPrompt_.empty()) {
        systemPrompt_ = "你是一个运行在手机本地的轻量级 AI 助手，名字叫 Pocket。"
                        "请用简洁、自然、口语化的中文回答，除非用户要求，不要写很长的列表。";
    }
    messages_.clear();
    messages_.emplace_back("system", systemPrompt_);
    return true;
}

void ChatEngine::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (llm_ != nullptr) {
        llm_->reset();
    }
    messages_.clear();
    if (!systemPrompt_.empty()) {
        messages_.emplace_back("system", systemPrompt_);
    }
}

bool ChatEngine::chat(const std::string& prompt, const DeltaFn& onDelta,
                      ChatStats* stats, std::string* error) {
    auto fail = [&](const std::string& message) {
        lastError_ = message;
        if (error != nullptr) {
            *error = message;
        }
        return false;
    };

    if (prompt.empty()) {
        return fail("输入内容为空");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (llm_ == nullptr) {
        return fail("模型尚未加载完成");
    }

    // Keep the context bounded: drop the oldest rounds, MNN re-prefills when the
    // cached prompt prefix no longer matches.
    if (messages_.size() > static_cast<size_t>(kMaxHistoryMessages)) {
        auto firstUser = messages_.begin() + 1;
        messages_.erase(firstUser, firstUser + (messages_.size() - kMaxHistoryMessages));
        llm_->reset();
    }

    messages_.emplace_back("user", prompt);

    const long long begin = nowMicros();
    long long firstDeltaUs = -1;
    Utf8Emitter emitter([&onDelta, &firstDeltaUs, begin](const std::string& text) {
        if (firstDeltaUs < 0) {
            firstDeltaUs = nowMicros() - begin;
        }
        if (onDelta) {
            onDelta(text);
        }
    });
    StreamBuffer buffer([&emitter](const char* data, size_t size) { emitter.push(data, size); });
    std::ostream stream(&buffer);

    llm_->response(messages_, &stream, nullptr, maxNewTokens_);
    emitter.flush();
    const long long elapsedUs = nowMicros() - begin;

    auto* context = llm_->getContext();
    if (context == nullptr) {
        return fail("推理上下文丢失");
    }
    if (context->status == MNN::Transformer::LlmStatus::INTERNAL_ERROR) {
        llm_->reset();
        messages_.pop_back();
        return fail("推理失败：引擎内部错误");
    }

    std::string answer = trim(context->generate_str);
    if (answer.empty()) {
        messages_.pop_back();
        return fail("模型没有返回内容");
    }
    messages_.emplace_back("assistant", answer);

    if (stats != nullptr) {
        stats->promptTokens = context->prompt_len;
        stats->outputTokens = static_cast<int>(context->output_tokens.size());
        stats->contextTokens = context->all_seq_len;
        stats->prefillMs = context->prefill_us / 1000;
        stats->decodeMs = context->decode_us / 1000;
        // MNN only fills ttfa_us on multimodal paths; measure it where it matters.
        stats->ttfaMs = context->ttfa_us > 0 ? context->ttfa_us / 1000
                                             : (firstDeltaUs > 0 ? firstDeltaUs / 1000 : 0);
        stats->totalMs = elapsedUs / 1000;
        stats->loadMs = loadMs_;
        stats->prefillTps =
            context->prefill_us > 0 ? (context->prefill_us / 1e6 > 0
                                           ? context->prompt_len / (context->prefill_us / 1e6)
                                           : 0.0)
                                    : 0.0;
        stats->decodeTps = context->decode_us > 0
                               ? stats->outputTokens / (context->decode_us / 1e6)
                               : 0.0;
    }
    return true;
}

std::string ChatEngine::describe() const {
    std::ostringstream os;
    os << "{\"model\":\"MNN-Pocket-0.3B\",\"loaded\":" << (llm_ != nullptr ? "true" : "false")
       << ",\"loadMs\":" << loadMs_ << ",\"maxNewTokens\":" << maxNewTokens_
       << ",\"systemPrompt\":\"" << jsonEscape(systemPrompt_) << "\"}";
    return os.str();
}

} // namespace pocket
