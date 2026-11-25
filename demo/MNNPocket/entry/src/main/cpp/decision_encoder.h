/*
 * DecisionEncoder - byte-faithful port of the request serializer used by
 * taobao-mnn/MNN-Pocket-0.3B-Dec (inference.py / inference_mnn.py).
 *
 * The model never sees a chat template: the state is rendered with the
 * training-time YAML-ish serializer, control sequences in user text are
 * escaped, and every question becomes an attention-isolated branch delimited
 * by reserved control tokens. Positions restart after the state, and the
 * dense mask is derived from branch segment ids. Any drift here changes the
 * logits, so this file mirrors the Python reference line by line.
 */

#ifndef POCKET_DECISION_ENCODER_H
#define POCKET_DECISION_ENCODER_H

#include <memory>
#include <string>
#include <vector>

#include "decision_json.h"
#include "tokenizer/tokenizer.hpp"

namespace pocket {

/** One normalized question: candidate keys, rendered option texts, instructions. */
struct DecisionQuestion {
    std::string id;
    std::string type; // choice | noul | score
    std::vector<std::string> keys;
    std::vector<std::string> options;
    std::string instructions; // already rendered
};

/** Fully encoded request, ready to be turned into the five graph inputs. */
struct EncodedRequest {
    std::vector<int> ids;
    std::vector<int> positions;
    std::vector<int> segments;
    std::vector<int> decidePositions;          // [Q]
    std::vector<std::vector<int>> optionEnds;  // [Q][Cq], unpadded
    std::vector<DecisionQuestion> questions;
};

class DecisionEncoder {
public:
    DecisionEncoder();
    ~DecisionEncoder();

    /**
     * Loads the MNN tokenizer and verifies the reserved control-token ids
     * against inference_config.json, exactly like the reference loader.
     */
    bool init(const std::string& tokenizerPath, const djson::Value& config, std::string* error);

    /** state + questions (insertion order preserved) -> encoded request. */
    bool encode(const djson::Value& state, const djson::Value& questions,
                EncodedRequest* out, std::string* error);

    /** Escapes control sequences and tokenizes, matching tokens() in the reference. */
    std::vector<int> tokenize(const std::string& text);

    int maxLength() const { return maxLength_; }
    bool ready() const { return tokenizer_ != nullptr; }

    /** Renders a JSON value with the training serializer (Python render()). */
    static std::string render(const djson::Value& value, int indent = 0);

    /** Builds the dense additive mask [L*L] from segment ids (row-major). */
    static void buildMask(const std::vector<int>& segments, float sentinel, std::vector<float>* out);

private:
    bool normalizeQuestion(const std::string& id, const djson::Value& value,
                           DecisionQuestion* out, std::string* error);

    std::unique_ptr<MNN::Transformer::Tokenizer> tokenizer_;
    int stateId_ = -1;
    int questionId_ = -1;
    int optionId_ = -1;
    int closeId_ = -1;
    int decideId_ = -1;
    int maxLength_ = 32768;
};

} // namespace pocket

#endif // POCKET_DECISION_ENCODER_H