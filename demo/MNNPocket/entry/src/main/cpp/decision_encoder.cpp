#include "decision_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace pocket {

namespace {

/** Python str(float): shortest round-trip repr, integral values keep ".0". */
std::string pyFloat(double v) {
    if (std::isfinite(v) && v == std::floor(v) && std::fabs(v) < 1e16) {
        char buf[48];
        std::snprintf(buf, sizeof(buf), "%.0f.0", v);
        return std::string(buf);
    }
    char buf[64];
    for (int prec = 1; prec <= 17; ++prec) {
        std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
        if (std::strtod(buf, nullptr) == v) {
            break;
        }
    }
    std::string s(buf);
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos) {
        s += ".0";
    }
    return s;
}

/** Python str() for the scalar JSON types the serializer accepts. */
std::string pyScalar(const djson::Value& value) {
    switch (value.kind) {
        case djson::Value::STR: return value.str;
        case djson::Value::BOOL: return value.boolean ? "True" : "False";
        case djson::Value::INT: {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%lld", value.integer);
            return std::string(buf);
        }
        case djson::Value::REAL: return pyFloat(value.real);
        default: return std::string();
    }
}

std::string lstrip(const std::string& s) {
    size_t begin = 0;
    while (begin < s.size()) {
        char c = s[begin];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v') {
            ++begin;
        } else {
            break;
        }
    }
    return s.substr(begin);
}

/**
 * Escapes reserved control sequences in user text:
 *   <|name|>  ->  <¦name¦>   (¦ = U+00A6, UTF-8 C2 A6)
 * Mirrors CONTROL_TOKEN = re.compile('<\\|([A-Za-z0-9_]+)\\|>').
 */
std::string escapeControl(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        if (text[i] == '<' && i + 1 < text.size() && text[i + 1] == '|') {
            size_t j = i + 2;
            while (j < text.size()) {
                char c = text[j];
                if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                    (c >= '0' && c <= '9') || c == '_') {
                    ++j;
                } else {
                    break;
                }
            }
            if (j > i + 2 && j + 1 < text.size() && text[j] == '|' && text[j + 1] == '>') {
                out += "<\xC2\xA6";
                out.append(text, i + 2, j - (i + 2));
                out += "\xC2\xA6>";
                i = j + 2;
                continue;
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

/** option(key, description) from the reference _question(). */
std::string renderOption(const std::string& key, const djson::Value* description) {
    if (description == nullptr || description->isNull() ||
        (description->isString() && description->str.empty())) {
        return key;
    }
    return key + ": " + DecisionEncoder::render(*description);
}

} // namespace

// ---------------------------------------------------------------------------
// render() - the training serializer
// ---------------------------------------------------------------------------

std::string DecisionEncoder::render(const djson::Value& value, int indent) {
    const std::string pad(static_cast<size_t>(indent) * 2, ' ');
    switch (value.kind) {
        case djson::Value::NUL:
            return std::string();
        case djson::Value::ARR: {
            std::string out;
            for (size_t i = 0; i < value.arr.size(); ++i) {
                if (i > 0) {
                    out += '\n';
                }
                out += pad + "- " + lstrip(render(value.arr[i], indent + 1));
            }
            return out;
        }
        case djson::Value::OBJ: {
            std::string out;
            for (size_t i = 0; i < value.obj.size(); ++i) {
                if (i > 0) {
                    out += '\n';
                }
                const auto& kv = value.obj[i];
                if (kv.second.isArray() || kv.second.isObject()) {
                    out += pad + kv.first + ":\n" + render(kv.second, indent + 1);
                } else {
                    out += pad + kv.first + ": " + render(kv.second, indent);
                }
            }
            return out;
        }
        default:
            return pyScalar(value);
    }
}

// ---------------------------------------------------------------------------
// mask
// ---------------------------------------------------------------------------

void DecisionEncoder::buildMask(const std::vector<int>& segments, float sentinel,
                                std::vector<float>* out) {
    const size_t length = segments.size();
    out->assign(length * length, 0.0f);
    for (size_t i = 0; i < length; ++i) {
        for (size_t j = 0; j < length; ++j) {
            // tril & (same segment | key in state segment) & key not padding;
            // the diagonal is always visible.
            const bool allow = (i == j) ||
                (j <= i && segments[j] != -1 &&
                 (segments[i] == segments[j] || segments[j] == 0));
            if (!allow) {
                (*out)[i * length + j] = sentinel;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DecisionEncoder
// ---------------------------------------------------------------------------

DecisionEncoder::DecisionEncoder() = default;
DecisionEncoder::~DecisionEncoder() = default;

bool DecisionEncoder::init(const std::string& tokenizerPath, const djson::Value& config,
                           std::string* error) {
    auto fail = [&](const std::string& message) {
        if (error != nullptr) {
            *error = message;
        }
        return false;
    };

    tokenizer_.reset(MNN::Transformer::Tokenizer::createTokenizer(tokenizerPath));
    if (tokenizer_ == nullptr) {
        return fail("无法加载决策 tokenizer: " + tokenizerPath);
    }

    const djson::Value* special = config.find("special_tokens");
    if (special == nullptr || !special->isObject()) {
        return fail("inference_config.json 缺少 special_tokens");
    }
    struct Marker { const char* token; int* id; };
    const Marker markers[] = {
        {"<|fim_prefix|>", &stateId_},
        {"<|fim_middle|>", &questionId_},
        {"<|box_start|>", &optionId_},
        {"<|box_end|>", &closeId_},
        {"<|fim_suffix|>", &decideId_},
    };
    for (const auto& marker : markers) {
        const djson::Value* idValue = special->find(marker.token);
        if (idValue == nullptr || idValue->kind != djson::Value::INT) {
            return fail(std::string("special_tokens 缺少控制 token: ") + marker.token);
        }
        *marker.id = static_cast<int>(idValue->integer);
        // The reference verifies tokenizer.token_to_id(token) == recorded id.
        // Encoding the marker must yield exactly that single id (special
        // tokens are split out atomically by the tokenizer).
        const std::vector<int> encoded = tokenizer_->encode(marker.token, false);
        if (encoded.size() != 1 || encoded[0] != *marker.id) {
            return fail(std::string("控制 token ID 校验失败: ") + marker.token +
                        " -> 期望 id " + std::to_string(*marker.id) + "，实际编码 " +
                        std::to_string(encoded.size()) + " 个 token");
        }
    }

    const djson::Value* maxLength = config.find("max_length");
    if (maxLength != nullptr && maxLength->kind == djson::Value::INT && maxLength->integer > 0) {
        maxLength_ = static_cast<int>(maxLength->integer);
    }
    return true;
}

std::vector<int> DecisionEncoder::tokenize(const std::string& text) {
    return tokenizer_->encode(escapeControl(text), false);
}

bool DecisionEncoder::normalizeQuestion(const std::string& id, const djson::Value& value,
                                        DecisionQuestion* out, std::string* error) {
    auto fail = [&](const std::string& message) {
        if (error != nullptr) {
            *error = "question '" + id + "': " + message;
        }
        return false;
    };

    if (!value.isObject()) {
        return fail("必须是对象");
    }
    const djson::Value* typeValue = value.find("type");
    if (typeValue == nullptr || !typeValue->isString()) {
        return fail("缺少 type");
    }
    out->id = id;
    out->type = typeValue->str;

    const djson::Value* criteria = value.find("criteria");
    const djson::Value* instructions = value.find("instructions");
    out->instructions = instructions != nullptr ? render(*instructions) : std::string();

    if (out->type == "choice") {
        if (criteria == nullptr || !criteria->isObject() || criteria->obj.empty()) {
            return fail("choice 的 criteria 必须是非空对象");
        }
        for (const auto& kv : criteria->obj) {
            out->keys.push_back(kv.first);
            out->options.push_back(renderOption(kv.first, &kv.second));
        }
    } else if (out->type == "noul") {
        if (criteria != nullptr && !criteria->isNull()) {
            if (!criteria->isObject()) {
                return fail("noul 的 criteria 必须是对象");
            }
            for (const auto& kv : criteria->obj) {
                if (kv.first != "false" && kv.first != "true") {
                    return fail("noul 的 criteria 只能包含 false / true");
                }
            }
        }
        const djson::Value* falseDesc = criteria != nullptr ? criteria->find("false") : nullptr;
        const djson::Value* trueDesc = criteria != nullptr ? criteria->find("true") : nullptr;
        out->keys = {"false", "true"};
        out->options = {renderOption("no", falseDesc), renderOption("yes", trueDesc)};
    } else if (out->type == "score") {
        if (criteria == nullptr || !criteria->isArray() || criteria->arr.empty()) {
            return fail("score 的 criteria 必须是非空有序列表");
        }
        for (size_t i = 0; i < criteria->arr.size(); ++i) {
            out->keys.push_back(std::to_string(i));
            out->options.push_back(render(criteria->arr[i]));
        }
    } else {
        return fail("type 必须是 choice、noul 或 score");
    }

    if (out->keys.empty() || out->keys.size() > 255) {
        return fail("候选数必须在 1..255 之间");
    }
    return true;
}

bool DecisionEncoder::encode(const djson::Value& state, const djson::Value& questions,
                             EncodedRequest* out, std::string* error) {
    auto fail = [&](const std::string& message) {
        if (error != nullptr) {
            *error = message;
        }
        return false;
    };

    if (tokenizer_ == nullptr) {
        return fail("编码器尚未初始化");
    }
    if (!questions.isObject() || questions.obj.empty()) {
        return fail("questions 必须是非空对象");
    }
    for (const auto& kv : questions.obj) {
        if (kv.first.empty()) {
            return fail("question ID 不能为空");
        }
    }

    *out = EncodedRequest();
    out->ids.push_back(stateId_);
    const std::vector<int> stateTokens = tokenize(render(state));
    out->ids.insert(out->ids.end(), stateTokens.begin(), stateTokens.end());
    out->segments.assign(out->ids.size(), 0);
    out->positions.resize(out->ids.size());
    for (size_t i = 0; i < out->positions.size(); ++i) {
        out->positions[i] = static_cast<int>(i);
    }
    const int stateLength = static_cast<int>(out->ids.size());

    int segment = 0;
    for (const auto& kv : questions.obj) {
        ++segment;
        DecisionQuestion question;
        if (!normalizeQuestion(kv.first, kv.second, &question, error)) {
            return false;
        }
        const int start = static_cast<int>(out->ids.size());
        std::vector<int> branch;
        branch.push_back(questionId_);
        const std::vector<int> instructionTokens = tokenize(question.instructions);
        branch.insert(branch.end(), instructionTokens.begin(), instructionTokens.end());
        std::vector<int> ends;
        for (const std::string& option : question.options) {
            branch.push_back(optionId_);
            const std::vector<int> optionTokens = tokenize(option);
            branch.insert(branch.end(), optionTokens.begin(), optionTokens.end());
            branch.push_back(closeId_);
            ends.push_back(start + static_cast<int>(branch.size()) - 1);
        }
        branch.push_back(decideId_);
        out->decidePositions.push_back(start + static_cast<int>(branch.size()) - 1);
        out->optionEnds.push_back(std::move(ends));
        out->ids.insert(out->ids.end(), branch.begin(), branch.end());
        out->segments.insert(out->segments.end(), branch.size(), segment);
        for (size_t i = 0; i < branch.size(); ++i) {
            out->positions.push_back(stateLength + static_cast<int>(i));
        }
        out->questions.push_back(std::move(question));
    }

    if (static_cast<int>(out->ids.size()) > maxLength_) {
        return fail("输入共 " + std::to_string(out->ids.size()) + " tokens，超过上限 " +
                    std::to_string(maxLength_) + "（不截断）");
    }
    return true;
}

} // namespace pocket