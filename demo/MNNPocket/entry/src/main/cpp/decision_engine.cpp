#include "decision_engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#include <MNN/Interpreter.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>

#include "llm_engine.h" // nowMicros, jsonEscape

namespace pocket {

namespace {

bool readFile(const std::string& path, std::string* out) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        return false;
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    *out = ss.str();
    return true;
}

/** FNV-1a over the raw mask bytes, used by the golden-vector parity test. */
std::string maskHash(const std::vector<float>& mask) {
    uint64_t hash = 14695981039346656037ULL;
    const auto* bytes = reinterpret_cast<const unsigned char*>(mask.data());
    const size_t count = mask.size() * sizeof(float);
    for (size_t i = 0; i < count; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(hash));
    return std::string(buf);
}

djson::Value intArray(const std::vector<int>& values) {
    djson::Value arr = djson::Value::array();
    for (int v : values) {
        arr.arr.push_back(djson::Value::int_(v));
    }
    return arr;
}

} // namespace

DecisionEngine::DecisionEngine() = default;

DecisionEngine::~DecisionEngine() {
    unload();
}

void DecisionEngine::unload() {
    if (module_ != nullptr) {
        MNN::Express::Module::destroy(module_);
        module_ = nullptr;
    }
    runtime_.reset();
}

bool DecisionEngine::load(const std::string& modelDir, const std::string& modelFile,
                          int threads, int maxTokens, std::string* error) {
    auto fail = [&](const std::string& message) {
        lastError_ = message;
        if (error != nullptr) {
            *error = message;
        }
        return false;
    };

    unload();
    const long long begin = nowMicros();

    std::string configText;
    if (!readFile(modelDir + "/inference_config.json", &configText)) {
        return fail("缺少 inference_config.json: " + modelDir);
    }
    djson::Value config;
    std::string parseError;
    if (!djson::parse(configText, &config, &parseError)) {
        return fail("inference_config.json 解析失败: " + parseError);
    }

    const djson::Value* temperature = config.find("temperature");
    if (temperature != nullptr &&
        (temperature->kind == djson::Value::REAL || temperature->kind == djson::Value::INT)) {
        temperature_ = temperature->kind == djson::Value::REAL ? temperature->real
                                                               : static_cast<double>(temperature->integer);
    }
    const djson::Value* sentinel = config.find("mask_sentinel");
    if (sentinel != nullptr && sentinel->kind == djson::Value::REAL) {
        maskSentinel_ = static_cast<float>(sentinel->real);
    }

    if (!encoder_.init(modelDir + "/tokenizer.mtok", config, &parseError)) {
        return fail(parseError);
    }

    const djson::Value* inputs = config.find("inputs");
    if (inputs == nullptr || !inputs->isObject() || inputs->obj.empty()) {
        return fail("inference_config.json 缺少 inputs 定义");
    }
    std::vector<std::string> inputNames;
    for (const auto& kv : inputs->obj) {
        inputNames.push_back(kv.first);
    }

    threads_ = threads >= 1 && threads <= 64 ? threads : 4;
    maxTokens_ = maxTokens > 0 ? maxTokens : 2048;
    modelFile_ = modelFile;

    // Match the reference runner: CPU, Normal precision, Normal memory.
    MNN::ScheduleConfig schedule;
    schedule.numThread = threads_;
    schedule.type = MNN_FORWARD_CPU;
    MNN::BackendConfig backend;
    backend.precision = MNN::BackendConfig::Precision_Normal;
    backend.memory = MNN::BackendConfig::Memory_Normal;
    schedule.backendConfig = &backend;
    runtime_.reset(MNN::Express::Executor::RuntimeManager::createRuntimeManager(schedule));

    MNN::Express::Module::Config moduleConfig;
    moduleConfig.dynamic = false;
    moduleConfig.shapeMutable = true;
    moduleConfig.rearrange = true;

    const std::string modelPath = modelDir + "/" + modelFile;
    module_ = MNN::Express::Module::load(inputNames, {"logits"}, modelPath.c_str(),
                                         runtime_, &moduleConfig);
    if (module_ == nullptr) {
        return fail("决策模型加载失败（文件损坏或内存不足）: " + modelPath);
    }

    loadMs_ = (nowMicros() - begin) / 1000;
    return true;
}

bool DecisionEngine::encodeRequest(const std::string& requestJson, EncodedRequest* encoded,
                                   double* temperature, std::string* error) {
    djson::Value request;
    if (!djson::parse(requestJson, &request, error)) {
        return false;
    }
    if (!request.isObject()) {
        *error = "请求必须是 JSON 对象";
        return false;
    }
    const djson::Value* state = request.find("state");
    const djson::Value* questions = request.find("questions");
    if (state == nullptr || questions == nullptr) {
        *error = "请求必须包含 state 和 questions";
        return false;
    }

    *temperature = temperature_;
    const djson::Value* override = request.find("temperature");
    if (override != nullptr && !override->isNull()) {
        double value = 0.0;
        if (override->kind == djson::Value::REAL) {
            value = override->real;
        } else if (override->kind == djson::Value::INT) {
            value = static_cast<double>(override->integer);
        } else {
            *error = "temperature 必须是有限正数";
            return false;
        }
        if (!std::isfinite(value) || value <= 0.0) {
            *error = "temperature 必须是有限正数";
            return false;
        }
        *temperature = value;
    }

    if (!encoder_.encode(*state, *questions, encoded, error)) {
        return false;
    }
    if (static_cast<int>(encoded->ids.size()) > maxTokens_) {
        *error = "输入共 " + std::to_string(encoded->ids.size()) + " tokens，超过端侧上限 " +
                 std::to_string(maxTokens_) + "（稠密掩码内存约束，可精简场景或在设置中调高）";
        return false;
    }
    return true;
}

bool DecisionEngine::forward(const EncodedRequest& encoded,
                             std::vector<std::vector<float>>* logits,
                             long long* forwardMs, std::string* error) {
    using namespace MNN::Express;
    const int length = static_cast<int>(encoded.ids.size());
    const int questionCount = static_cast<int>(encoded.questions.size());
    int candidateMax = 1;
    for (const auto& question : encoded.questions) {
        candidateMax = std::max(candidateMax, static_cast<int>(question.keys.size()));
    }

    auto idsVar = _Input({1, length}, NCHW, halide_type_of<int32_t>());
    std::memcpy(idsVar->writeMap<int32_t>(), encoded.ids.data(), length * sizeof(int32_t));
    auto posVar = _Input({1, length}, NCHW, halide_type_of<int32_t>());
    std::memcpy(posVar->writeMap<int32_t>(), encoded.positions.data(), length * sizeof(int32_t));

    std::vector<float> mask;
    DecisionEncoder::buildMask(encoded.segments, maskSentinel_, &mask);
    auto maskVar = _Input({1, 1, length, length}, NCHW, halide_type_of<float>());
    std::memcpy(maskVar->writeMap<float>(), mask.data(), mask.size() * sizeof(float));

    auto decideVar = _Input({1, questionCount}, NCHW, halide_type_of<int32_t>());
    std::memcpy(decideVar->writeMap<int32_t>(), encoded.decidePositions.data(),
                questionCount * sizeof(int32_t));

    auto optionVar = _Input({1, questionCount, candidateMax}, NCHW, halide_type_of<int32_t>());
    auto* optionData = optionVar->writeMap<int32_t>();
    std::memset(optionData, 0, questionCount * candidateMax * sizeof(int32_t));
    for (int q = 0; q < questionCount; ++q) {
        const auto& ends = encoded.optionEnds[q];
        for (size_t c = 0; c < ends.size(); ++c) {
            optionData[q * candidateMax + static_cast<int>(c)] = ends[c];
        }
    }

    const long long begin = nowMicros();
    const std::vector<VARP> outputs = module_->onForward({idsVar, posVar, maskVar, decideVar, optionVar});
    if (outputs.empty() || outputs[0] == nullptr) {
        *error = "决策前向失败：无输出";
        return false;
    }
    const auto* values = outputs[0]->readMap<float>();
    if (values == nullptr) {
        *error = "决策前向失败：无法读取 logits";
        return false;
    }
    logits->resize(questionCount);
    for (int q = 0; q < questionCount; ++q) {
        const int count = static_cast<int>(encoded.questions[q].keys.size());
        (*logits)[q].assign(values + q * candidateMax, values + q * candidateMax + count);
        for (float v : (*logits)[q]) {
            if (!std::isfinite(v)) {
                *error = "决策 logits 出现非有限值（检查精度设置）";
                return false;
            }
        }
    }
    *forwardMs = (nowMicros() - begin) / 1000;
    return true;
}

bool DecisionEngine::predict(const std::string& requestJson, std::string* resultJson,
                             DecisionStats* stats, std::string* error) {
    if (module_ == nullptr) {
        *error = "决策模型尚未加载";
        return false;
    }

    EncodedRequest encoded;
    double temperature = temperature_;
    const long long encodeBegin = nowMicros();
    if (!encodeRequest(requestJson, &encoded, &temperature, error)) {
        return false;
    }
    const long long encodeMs = (nowMicros() - encodeBegin) / 1000;

    std::vector<std::vector<float>> logits;
    long long forwardMs = 0;
    if (!forward(encoded, &logits, &forwardMs, error)) {
        return false;
    }

    djson::Value answers = djson::Value::object();
    for (size_t q = 0; q < encoded.questions.size(); ++q) {
        const DecisionQuestion& question = encoded.questions[q];
        const std::vector<float>& scores = logits[q];
        const float maximum = *std::max_element(scores.begin(), scores.end());
        std::vector<double> weights(scores.size());
        double total = 0.0;
        for (size_t i = 0; i < scores.size(); ++i) {
            weights[i] = std::exp((scores[i] - maximum) / temperature);
            total += weights[i];
        }
        int winner = 0;
        for (size_t i = 1; i < scores.size(); ++i) {
            if (scores[i] > scores[winner]) {
                winner = static_cast<int>(i);
            }
        }

        djson::Value answer = djson::Value::object();
        answer.set("type", djson::Value::string(question.type));
        djson::Value probabilities = djson::Value::object();
        for (size_t i = 0; i < scores.size(); ++i) {
            probabilities.set(question.keys[i], djson::Value::real_(weights[i] / total));
        }
        answer.set("probabilities", probabilities);
        if (question.type == "choice") {
            answer.set("choice", djson::Value::string(question.keys[winner]));
        } else if (question.type == "noul") {
            answer.set("decision", djson::Value::boolean_(winner == 1));
            answer.set("noul", djson::Value::real_(weights[1] / total));
        } else {
            double score = 0.0;
            for (size_t i = 0; i < scores.size(); ++i) {
                score += static_cast<double>(i) * (weights[i] / total);
            }
            answer.set("score", djson::Value::real_(score));
            djson::Value legend = djson::Value::object();
            for (size_t i = 0; i < question.keys.size(); ++i) {
                legend.set(question.keys[i], djson::Value::string(question.options[i]));
            }
            answer.set("legend", legend);
        }
        answers.set(question.id, std::move(answer));
    }

    djson::Value result = djson::Value::object();
    result.set("answers", std::move(answers));
    result.set("input_tokens", djson::Value::int_(static_cast<long long>(encoded.ids.size())));
    result.set("temperature", djson::Value::real_(temperature));
    *resultJson = djson::write(result);

    if (stats != nullptr) {
        stats->loadMs = loadMs_;
        stats->encodeMs = encodeMs;
        stats->forwardMs = forwardMs;
        stats->inputTokens = static_cast<int>(encoded.ids.size());
        stats->questionCount = static_cast<int>(encoded.questions.size());
        stats->temperature = temperature;
    }
    return true;
}

bool DecisionEngine::encodeDump(const std::string& requestJson, std::string* dumpJson,
                                std::string* error) {
    EncodedRequest encoded;
    double temperature = temperature_;
    if (!encodeRequest(requestJson, &encoded, &temperature, error)) {
        return false;
    }
    std::vector<float> mask;
    DecisionEncoder::buildMask(encoded.segments, maskSentinel_, &mask);

    djson::Value dump = djson::Value::object();
    dump.set("ids", intArray(encoded.ids));
    dump.set("pos", intArray(encoded.positions));
    dump.set("seg", intArray(encoded.segments));
    dump.set("decide_idx", intArray(encoded.decidePositions));
    djson::Value optIdx = djson::Value::array();
    for (const auto& ends : encoded.optionEnds) {
        optIdx.arr.push_back(intArray(ends));
    }
    dump.set("opt_idx", std::move(optIdx));
    dump.set("mask_hash", djson::Value::string(maskHash(mask)));
    dump.set("length", djson::Value::int_(static_cast<long long>(encoded.ids.size())));
    *dumpJson = djson::write(dump);
    return true;
}

std::string DecisionEngine::describe() const {
    std::ostringstream os;
    os << "{\"model\":\"MNN-Pocket-0.3B-Dec\",\"variant\":\"" << jsonEscape(modelFile_)
       << "\",\"loaded\":" << (module_ != nullptr ? "true" : "false")
       << ",\"loadMs\":" << loadMs_
       << ",\"maxTokens\":" << maxTokens_
       << ",\"threads\":" << threads_
       << ",\"temperature\":" << temperature_ << "}";
    return os.str();
}

} // namespace pocket