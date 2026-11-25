/*
 * DecisionEngine - single-forward inference for the MNN decision graph.
 *
 * Unlike ChatEngine this does NOT use the MNN LLM runtime: the converted
 * model is a plain 5-input / 1-output graph (input_ids, position_ids,
 * attention_mask, decide_positions, option_positions -> logits[Q, C]).
 * Temperature softmax and the answer semantics live outside the graph,
 * mirroring inference_mnn.py exactly.
 */

#ifndef POCKET_DECISION_ENGINE_H
#define POCKET_DECISION_ENGINE_H

#include <memory>
#include <string>

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>

#include "decision_encoder.h"
#include "decision_json.h"

namespace pocket {

struct DecisionStats {
    long long loadMs = 0;
    long long encodeMs = 0;
    long long forwardMs = 0; // includes resize on shape change + forward
    int inputTokens = 0;
    int questionCount = 0;
    double temperature = 1.0;
};

class DecisionEngine {
public:
    DecisionEngine();
    ~DecisionEngine();

    DecisionEngine(const DecisionEngine&) = delete;
    DecisionEngine& operator=(const DecisionEngine&) = delete;

    /**
     * @param modelDir  directory holding the .mnn graph, tokenizer.mtok and
     *                  inference_config.json
     * @param modelFile graph file name inside modelDir
     * @param maxTokens device-side request cap (dense L*L mask memory); the
     *                  checkpoint limit still applies on top
     */
    bool load(const std::string& modelDir, const std::string& modelFile,
              int threads, int maxTokens, std::string* error);

    /** requestJson = {"state":..., "questions":{...}, "temperature"?:number} */
    bool predict(const std::string& requestJson, std::string* resultJson,
                 DecisionStats* stats, std::string* error);

    /** Encoding-only dump for golden-vector parity tests (no forward pass). */
    bool encodeDump(const std::string& requestJson, std::string* dumpJson, std::string* error);

    void unload();
    bool loaded() const { return module_ != nullptr; }
    long long loadMs() const { return loadMs_; }
    std::string describe() const;

private:
    bool encodeRequest(const std::string& requestJson, EncodedRequest* encoded,
                       double* temperature, std::string* error);
    bool forward(const EncodedRequest& encoded, std::vector<std::vector<float>>* logits,
                 long long* forwardMs, std::string* error);

    DecisionEncoder encoder_;
    MNN::Express::Module* module_ = nullptr;
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> runtime_;
    std::string modelFile_;
    std::string lastError_;
    float maskSentinel_ = -3.3895313892515355e38f;
    double temperature_ = 1.1111832893308136;
    int threads_ = 4;
    int maxTokens_ = 2048;
    long long loadMs_ = 0;
};

} // namespace pocket

#endif // POCKET_DECISION_ENGINE_H