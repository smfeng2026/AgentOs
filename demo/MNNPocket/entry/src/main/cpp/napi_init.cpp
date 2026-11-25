/*
 * libllmbridge - the ArkTS <-> C++ boundary for the Pocket chat screen.
 *
 * The MNN LLM runtime is not thread safe and blocks for seconds per turn, so a
 * single dedicated worker thread owns the engine. JavaScript never calls into
 * MNN directly: it posts tasks onto a queue and receives progress through one
 * thread safe function (see setEventCallback), which keeps the UI thread free
 * and lets tokens appear while they are being generated.
 */

#include <napi/native_api.h>
#include <hilog/log.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

#include "decision_engine.h"
#include "llm_engine.h"

#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0x3200
#define LOG_TAG "PocketLLM"

#define POCKET_LOGI(fmt, ...) OH_LOG_Print(LOG_APP, LOG_INFO, LOG_DOMAIN, LOG_TAG, fmt, ##__VA_ARGS__)
#define POCKET_LOGE(fmt, ...) OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_DOMAIN, LOG_TAG, fmt, ##__VA_ARGS__)

namespace {

// ---------------------------------------------------------------------------
// worker task queue
// ---------------------------------------------------------------------------

struct Task {
    enum Kind { INIT, CHAT, RESET, DEC_INIT, DECIDE, DEC_UNLOAD } kind = INIT;
    std::string configPath;
    std::string tmpDir;
    std::string prompt;
    int maxTokens = 512;
    // decision tasks
    std::string modelDir;
    std::string modelFile;
    std::string requestJson;
    int threads = 4;
    int decMaxTokens = 2048;
};

std::mutex g_queueMutex;
std::condition_variable g_queueCv;
std::deque<Task> g_tasks;
bool g_shouldStop = false;
std::thread g_worker;
bool g_workerStarted = false;

pocket::ChatEngine* g_engine = nullptr; // owned by the worker thread
std::atomic<bool> g_ready{false};
std::atomic<bool> g_generating{false};

pocket::DecisionEngine* g_decEngine = nullptr; // owned by the worker thread
std::atomic<bool> g_decReady{false};
std::atomic<bool> g_decRunning{false};

std::mutex g_infoMutex;
std::string g_infoJson = "{\"model\":\"MNN-Pocket-0.3B\",\"loaded\":false}";
std::string g_decInfoJson = "{\"model\":\"MNN-Pocket-0.3B-Dec\",\"loaded\":false}";

napi_threadsafe_function g_tsfn = nullptr;

// ---------------------------------------------------------------------------
// event delivery
// ---------------------------------------------------------------------------

struct EventPayload {
    std::string name;
    std::string body;
};

void CallJs(napi_env env, napi_value jsCallback, void* /*context*/, void* data) {
    auto* payload = static_cast<EventPayload*>(data);
    if (payload == nullptr) {
        return;
    }
    if (env != nullptr && jsCallback != nullptr) {
        napi_value argv[2];
        napi_value undefined = nullptr;
        napi_get_undefined(env, &undefined);
        napi_create_string_utf8(env, payload->name.c_str(), NAPI_AUTO_LENGTH, &argv[0]);
        napi_create_string_utf8(env, payload->body.c_str(), NAPI_AUTO_LENGTH, &argv[1]);
        napi_call_function(env, undefined, jsCallback, 2, argv, nullptr);
    }
    delete payload;
}

/** Safe to call from any thread. */
void emitEvent(const std::string& name, const std::string& body) {
    if (g_tsfn == nullptr) {
        return;
    }
    auto* payload = new EventPayload{name, body};
    napi_status status = napi_call_threadsafe_function(g_tsfn, payload, napi_tsfn_nonblocking);
    if (status != napi_ok) {
        // The JS side is gone (page destroyed): drop the event.
        delete payload;
    }
}

// ---------------------------------------------------------------------------
// worker handlers
// ---------------------------------------------------------------------------

void handleInit(const Task& task) {
    emitEvent("init_progress", "{\"stage\":\"创建推理引擎\"}");
    if (g_engine != nullptr) {
        delete g_engine;
        g_engine = nullptr;
        g_ready = false;
    }
    g_engine = new pocket::ChatEngine();
    std::string error;
    if (!g_engine->load(task.configPath, task.tmpDir, task.maxTokens, &error)) {
        POCKET_LOGE("model load failed: %{public}s", error.c_str());
        emitEvent("error", "{\"stage\":\"init\",\"message\":\"" + pocket::jsonEscape(error) + "\"}");
        return;
    }
    {
        std::lock_guard<std::mutex> lock(g_infoMutex);
        g_infoJson = g_engine->describe();
    }
    g_ready = true;
    POCKET_LOGI("model ready, load %{public}lld ms", g_engine->loadMs());
    emitEvent("ready", g_engine->describe());
}

void handleChat(const Task& task) {
    if (g_engine == nullptr || !g_engine->loaded()) {
        emitEvent("error",
                  "{\"stage\":\"chat\",\"message\":\"" +
                      pocket::jsonEscape("模型尚未加载完成，请稍候") + "\"}");
        return;
    }
    g_generating = true;
    emitEvent("start", "{\"prompt\":\"" + pocket::jsonEscape(task.prompt) + "\"}");

    pocket::ChatStats stats;
    std::string error;
    std::string collected;
    auto onDelta = [&collected](const std::string& text) {
        collected += text;
        emitEvent("delta", "{\"text\":\"" + pocket::jsonEscape(text) + "\"}");
    };
    const bool ok = g_engine->chat(task.prompt, onDelta, &stats, &error);
    g_generating = false;

    if (!ok) {
        POCKET_LOGE("chat failed: %{public}s", error.c_str());
        emitEvent("error", "{\"stage\":\"chat\",\"message\":\"" + pocket::jsonEscape(error) + "\"}");
        return;
    }
    POCKET_LOGI("turn done: prompt=%{public}d out=%{public}d prefill=%{public}lldms decode=%{public}lldms",
                stats.promptTokens, stats.outputTokens, stats.prefillMs, stats.decodeMs);

    std::string body = stats.toJson();
    body.pop_back(); // drop trailing '}'
    body += ",\"text\":\"" + pocket::jsonEscape(collected) + "\"}";
    emitEvent("done", body);
}

void handleReset() {
    if (g_engine != nullptr) {
        g_engine->reset();
    }
    emitEvent("reset_done", "{}");
}

// ---------------------------------------------------------------------------
// decision handlers (same worker thread: chat and decide never run at once)
// ---------------------------------------------------------------------------

void handleDecInit(const Task& task) {
    emitEvent("dec_progress", "{\"stage\":\"创建决策引擎\"}");
    if (g_decEngine != nullptr) {
        delete g_decEngine;
        g_decEngine = nullptr;
        g_decReady = false;
    }
    g_decEngine = new pocket::DecisionEngine();
    std::string error;
    if (!g_decEngine->load(task.modelDir, task.modelFile, task.threads, task.decMaxTokens, &error)) {
        POCKET_LOGE("decision model load failed: %{public}s", error.c_str());
        emitEvent("dec_error", "{\"stage\":\"init\",\"message\":\"" + pocket::jsonEscape(error) + "\"}");
        return;
    }
    {
        std::lock_guard<std::mutex> lock(g_infoMutex);
        g_decInfoJson = g_decEngine->describe();
    }
    g_decReady = true;
    POCKET_LOGI("decision model ready, load %{public}lld ms", g_decEngine->loadMs());
    emitEvent("dec_ready", g_decEngine->describe());
}

void handleDecide(const Task& task) {
    if (g_decEngine == nullptr || !g_decEngine->loaded()) {
        emitEvent("dec_error",
                  "{\"stage\":\"decide\",\"message\":\"" +
                      pocket::jsonEscape("决策模型尚未加载完成，请稍候") + "\"}");
        return;
    }
    g_decRunning = true;
    emitEvent("dec_start", "{}");

    std::string result;
    pocket::DecisionStats stats;
    std::string error;
    const bool ok = g_decEngine->predict(task.requestJson, &result, &stats, &error);
    g_decRunning = false;

    if (!ok) {
        POCKET_LOGE("decide failed: %{public}s", error.c_str());
        emitEvent("dec_error", "{\"stage\":\"decide\",\"message\":\"" + pocket::jsonEscape(error) + "\"}");
        return;
    }
    POCKET_LOGI("decide done: input=%{public}d tok forward=%{public}lld ms",
                stats.inputTokens, stats.forwardMs);

    // Merge the engine timing report into the result JSON (which already ends
    // with '}'), mirroring how chat 'done' carries its stats.
    std::string body = result;
    body.pop_back();
    char extra[256];
    std::snprintf(extra, sizeof(extra),
                  ",\"encodeMs\":%lld,\"forwardMs\":%lld,\"loadMs\":%lld,\"questionCount\":%d}",
                  stats.encodeMs, stats.forwardMs, stats.loadMs, stats.questionCount);
    body += extra;
    emitEvent("dec_done", body);
}

void handleDecUnload() {
    if (g_decEngine != nullptr) {
        g_decEngine->unload();
        g_decReady = false;
        std::lock_guard<std::mutex> lock(g_infoMutex);
        g_decInfoJson = g_decEngine->describe();
    }
    emitEvent("dec_unloaded", "{}");
}

void workerMain() {
    for (;;) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(g_queueMutex);
            g_queueCv.wait(lock, [] { return g_shouldStop || !g_tasks.empty(); });
            if (g_tasks.empty()) {
                if (g_shouldStop) {
                    break;
                }
                continue;
            }
            task = std::move(g_tasks.front());
            g_tasks.pop_front();
        }
        switch (task.kind) {
            case Task::INIT:
                handleInit(task);
                break;
            case Task::CHAT:
                handleChat(task);
                break;
            case Task::RESET:
                handleReset();
                break;
            case Task::DEC_INIT:
                handleDecInit(task);
                break;
            case Task::DECIDE:
                handleDecide(task);
                break;
            case Task::DEC_UNLOAD:
                handleDecUnload();
                break;
        }
    }
    // Free the models on the thread that created them.
    delete g_engine;
    g_engine = nullptr;
    g_ready = false;
    delete g_decEngine;
    g_decEngine = nullptr;
    g_decReady = false;
}

void ensureWorker() {
    if (!g_workerStarted) {
        g_shouldStop = false;
        g_worker = std::thread(workerMain);
        g_workerStarted = true;
    }
}

void enqueue(Task&& task) {
    {
        std::lock_guard<std::mutex> lock(g_queueMutex);
        g_tasks.push_back(std::move(task));
    }
    g_queueCv.notify_one();
}

// ---------------------------------------------------------------------------
// argument helpers
// ---------------------------------------------------------------------------

bool readString(napi_env env, napi_value value, std::string* out) {
    size_t length = 0;
    if (napi_get_value_string_utf8(env, value, nullptr, 0, &length) != napi_ok) {
        return false;
    }
    out->resize(length);
    size_t copied = 0;
    // napi writes a terminating '\0' as well, so pass an extra byte of room.
    std::string buffer(length + 1, '\0');
    if (napi_get_value_string_utf8(env, value, buffer.data(), length + 1, &copied) != napi_ok) {
        return false;
    }
    buffer.resize(copied);
    *out = buffer;
    return true;
}

napi_value makeBool(napi_env env, bool value) {
    napi_value result = nullptr;
    napi_get_boolean(env, value, &result);
    return result;
}

// ---------------------------------------------------------------------------
// NAPI entry points
// ---------------------------------------------------------------------------

napi_value setEventCallback(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc < 1) {
        return nullptr;
    }
    napi_valuetype type;
    napi_typeof(env, args[0], &type);
    if (type != napi_function) {
        return nullptr;
    }
    if (g_tsfn != nullptr) {
        napi_release_threadsafe_function(g_tsfn, napi_tsfn_abort);
        g_tsfn = nullptr;
    }
    napi_value resourceName = nullptr;
    napi_create_string_utf8(env, "PocketLlmBridge", NAPI_AUTO_LENGTH, &resourceName);
    napi_create_threadsafe_function(env, args[0], nullptr, resourceName, 0, 1, nullptr, nullptr,
                                    nullptr, CallJs, &g_tsfn);
    return nullptr;
}

napi_value initModel(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value args[3] = {nullptr, nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    Task task;
    task.kind = Task::INIT;
    if (argc >= 1) {
        readString(env, args[0], &task.configPath);
    }
    if (argc >= 2) {
        readString(env, args[1], &task.tmpDir);
    }
    if (argc >= 3) {
        int32_t maxTokens = 512;
        napi_get_value_int32(env, args[2], &maxTokens);
        task.maxTokens = maxTokens;
    }
    POCKET_LOGI("init requested: %{public}s", task.configPath.c_str());
    ensureWorker();
    enqueue(std::move(task));
    return nullptr;
}

napi_value chat(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    Task task;
    task.kind = Task::CHAT;
    if (argc >= 1) {
        readString(env, args[0], &task.prompt);
    }
    if (task.prompt.empty()) {
        return nullptr;
    }
    if (g_generating) {
        emitEvent("error", "{\"stage\":\"chat\",\"message\":\"上一轮回答仍在生成中\"}");
        return nullptr;
    }
    if (g_decRunning) {
        emitEvent("error", "{\"stage\":\"chat\",\"message\":\"决策推理进行中，请稍候\"}");
        return nullptr;
    }
    enqueue(std::move(task));
    return nullptr;
}

napi_value resetSession(napi_env env, napi_callback_info /*info*/) {
    if (g_engine != nullptr) {
        Task task;
        task.kind = Task::RESET;
        enqueue(std::move(task));
    }
    return nullptr;
}

napi_value isReady(napi_env env, napi_callback_info /*info*/) {
    return makeBool(env, g_ready.load());
}

napi_value isGenerating(napi_env env, napi_callback_info /*info*/) {
    return makeBool(env, g_generating.load());
}

napi_value modelInfo(napi_env env, napi_callback_info /*info*/) {
    std::string json;
    {
        std::lock_guard<std::mutex> lock(g_infoMutex);
        json = g_infoJson;
    }
    napi_value result = nullptr;
    napi_create_string_utf8(env, json.c_str(), NAPI_AUTO_LENGTH, &result);
    return result;
}

napi_value shutdown(napi_env env, napi_callback_info /*info*/) {
    {
        std::lock_guard<std::mutex> lock(g_queueMutex);
        g_shouldStop = true;
    }
    g_queueCv.notify_all();
    if (g_workerStarted && g_worker.joinable()) {
        g_worker.join();
        g_workerStarted = false;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// decision entry points
// ---------------------------------------------------------------------------

napi_value initDecision(napi_env env, napi_callback_info info) {
    size_t argc = 4;
    napi_value args[4] = {nullptr, nullptr, nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    Task task;
    task.kind = Task::DEC_INIT;
    if (argc >= 1) {
        readString(env, args[0], &task.modelDir);
    }
    task.modelFile = "decision-w8-q8-embedding.mnn";
    if (argc >= 2) {
        readString(env, args[1], &task.modelFile);
    }
    if (argc >= 3) {
        int32_t threads = 4;
        napi_get_value_int32(env, args[2], &threads);
        task.threads = threads;
    }
    if (argc >= 4) {
        int32_t maxTokens = 2048;
        napi_get_value_int32(env, args[3], &maxTokens);
        task.decMaxTokens = maxTokens;
    }
    POCKET_LOGI("decision init requested: %{public}s", task.modelDir.c_str());
    ensureWorker();
    enqueue(std::move(task));
    return nullptr;
}

napi_value decide(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    Task task;
    task.kind = Task::DECIDE;
    if (argc >= 1) {
        readString(env, args[0], &task.requestJson);
    }
    if (task.requestJson.empty()) {
        return nullptr;
    }
    if (g_decRunning) {
        emitEvent("dec_error", "{\"stage\":\"decide\",\"message\":\"上一次决策仍在推理中\"}");
        return nullptr;
    }
    if (g_generating) {
        emitEvent("dec_error", "{\"stage\":\"decide\",\"message\":\"对话生成进行中，请稍候\"}");
        return nullptr;
    }
    ensureWorker();
    enqueue(std::move(task));
    return nullptr;
}

napi_value unloadDecision(napi_env env, napi_callback_info /*info*/) {
    if (g_decEngine != nullptr && !g_decRunning) {
        Task task;
        task.kind = Task::DEC_UNLOAD;
        ensureWorker();
        enqueue(std::move(task));
    }
    return nullptr;
}

napi_value isDecisionReady(napi_env env, napi_callback_info /*info*/) {
    return makeBool(env, g_decReady.load());
}

napi_value decisionInfo(napi_env env, napi_callback_info /*info*/) {
    std::string json;
    {
        std::lock_guard<std::mutex> lock(g_infoMutex);
        json = g_decInfoJson;
    }
    napi_value result = nullptr;
    napi_create_string_utf8(env, json.c_str(), NAPI_AUTO_LENGTH, &result);
    return result;
}

napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor desc[] = {
        {"setEventCallback", nullptr, setEventCallback, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"initModel", nullptr, initModel, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"chat", nullptr, chat, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"resetSession", nullptr, resetSession, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"isReady", nullptr, isReady, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"isGenerating", nullptr, isGenerating, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"modelInfo", nullptr, modelInfo, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"shutdown", nullptr, shutdown, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"initDecision", nullptr, initDecision, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"decide", nullptr, decide, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"unloadDecision", nullptr, unloadDecision, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"isDecisionReady", nullptr, isDecisionReady, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"decisionInfo", nullptr, decisionInfo, nullptr, nullptr, nullptr, napi_default, nullptr},
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}

napi_module module = {
    1,               // nm_version
    0,               // nm_flags
    nullptr,         // nm_filename
    Init,            // nm_register_func
    "llmbridge",     // nm_modname
    nullptr,         // nm_priv
    {nullptr},       // reserved
};

} // namespace

extern "C" __attribute__((constructor)) void RegisterLlmBridgeModule() {
    napi_module_register(&module);
}
