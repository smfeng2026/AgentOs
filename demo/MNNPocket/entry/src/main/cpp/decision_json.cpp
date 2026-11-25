#include "decision_json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace pocket {
namespace djson {

const Value* Value::find(const std::string& key) const {
    if (kind != OBJ) {
        return nullptr;
    }
    for (const auto& kv : obj) {
        if (kv.first == key) {
            return &kv.second;
        }
    }
    return nullptr;
}

Value* Value::find(const std::string& key) {
    if (kind != OBJ) {
        return nullptr;
    }
    for (auto& kv : obj) {
        if (kv.first == key) {
            return &kv.second;
        }
    }
    return nullptr;
}

void Value::set(const std::string& key, Value value) {
    for (auto& kv : obj) {
        if (kv.first == key) {
            kv.second = std::move(value);
            return;
        }
    }
    obj.emplace_back(key, std::move(value));
}

// ---------------------------------------------------------------------------
// parser
// ---------------------------------------------------------------------------

namespace {

class Parser {
public:
    Parser(const std::string& text) : text_(text) {}

    bool parse(Value* out) {
        skipWs();
        if (!parseValue(out, 0)) {
            return false;
        }
        skipWs();
        if (pos_ != text_.size()) {
            return fail("trailing characters after JSON value");
        }
        return true;
    }

    const std::string& error() const { return error_; }

private:
    bool fail(const std::string& message) {
        if (error_.empty()) {
            error_ = message + " (at offset " + std::to_string(pos_) + ")";
        }
        return false;
    }

    void skipWs() {
        while (pos_ < text_.size()) {
            char c = text_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    char peek() const { return pos_ < text_.size() ? text_[pos_] : '\0'; }

    bool parseValue(Value* out, int depth) {
        if (depth > 64) {
            return fail("JSON nesting too deep");
        }
        skipWs();
        switch (peek()) {
            case '{': return parseObject(out, depth);
            case '[': return parseArray(out, depth);
            case '"': {
                out->kind = Value::STR;
                return parseString(&out->str);
            }
            case 't':
                if (text_.compare(pos_, 4, "true") == 0) {
                    pos_ += 4;
                    *out = Value::boolean_(true);
                    return true;
                }
                return fail("invalid literal");
            case 'f':
                if (text_.compare(pos_, 5, "false") == 0) {
                    pos_ += 5;
                    *out = Value::boolean_(false);
                    return true;
                }
                return fail("invalid literal");
            case 'n':
                if (text_.compare(pos_, 4, "null") == 0) {
                    pos_ += 4;
                    *out = Value();
                    return true;
                }
                return fail("invalid literal");
            default:
                return parseNumber(out);
        }
    }

    bool parseObject(Value* out, int depth) {
        *out = Value::object();
        ++pos_; // '{'
        skipWs();
        if (peek() == '}') {
            ++pos_;
            return true;
        }
        for (;;) {
            skipWs();
            if (peek() != '"') {
                return fail("expected object key string");
            }
            std::string key;
            if (!parseString(&key)) {
                return false;
            }
            skipWs();
            if (peek() != ':') {
                return fail("expected ':' after object key");
            }
            ++pos_;
            Value value;
            if (!parseValue(&value, depth + 1)) {
                return false;
            }
            out->set(key, std::move(value));
            skipWs();
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            if (peek() == '}') {
                ++pos_;
                return true;
            }
            return fail("expected ',' or '}' in object");
        }
    }

    bool parseArray(Value* out, int depth) {
        *out = Value::array();
        ++pos_; // '['
        skipWs();
        if (peek() == ']') {
            ++pos_;
            return true;
        }
        for (;;) {
            Value value;
            if (!parseValue(&value, depth + 1)) {
                return false;
            }
            out->arr.push_back(std::move(value));
            skipWs();
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            if (peek() == ']') {
                ++pos_;
                return true;
            }
            return fail("expected ',' or ']' in array");
        }
    }

    static void appendUtf8(std::string* out, unsigned int cp) {
        if (cp <= 0x7F) {
            out->push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FF) {
            out->push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp <= 0xFFFF) {
            out->push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out->push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    bool parseHex4(unsigned int* out) {
        if (pos_ + 4 > text_.size()) {
            return fail("truncated \\u escape");
        }
        unsigned int value = 0;
        for (int i = 0; i < 4; ++i) {
            char c = text_[pos_++];
            value <<= 4;
            if (c >= '0' && c <= '9') {
                value |= static_cast<unsigned int>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                value |= static_cast<unsigned int>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                value |= static_cast<unsigned int>(c - 'A' + 10);
            } else {
                return fail("invalid \\u escape");
            }
        }
        *out = value;
        return true;
    }

    bool parseString(std::string* out) {
        ++pos_; // opening quote
        out->clear();
        for (;;) {
            if (pos_ >= text_.size()) {
                return fail("unterminated string");
            }
            char c = text_[pos_++];
            if (c == '"') {
                return true;
            }
            if (c == '\\') {
                if (pos_ >= text_.size()) {
                    return fail("unterminated escape");
                }
                char e = text_[pos_++];
                switch (e) {
                    case '"': out->push_back('"'); break;
                    case '\\': out->push_back('\\'); break;
                    case '/': out->push_back('/'); break;
                    case 'b': out->push_back('\b'); break;
                    case 'f': out->push_back('\f'); break;
                    case 'n': out->push_back('\n'); break;
                    case 'r': out->push_back('\r'); break;
                    case 't': out->push_back('\t'); break;
                    case 'u': {
                        unsigned int cp = 0;
                        if (!parseHex4(&cp)) {
                            return false;
                        }
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            if (pos_ + 2 <= text_.size() && text_[pos_] == '\\' && text_[pos_ + 1] == 'u') {
                                pos_ += 2;
                                unsigned int low = 0;
                                if (!parseHex4(&low)) {
                                    return false;
                                }
                                if (low >= 0xDC00 && low <= 0xDFFF) {
                                    cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                                } else {
                                    return fail("invalid surrogate pair");
                                }
                            } else {
                                return fail("lone high surrogate");
                            }
                        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                            return fail("lone low surrogate");
                        }
                        appendUtf8(out, cp);
                        break;
                    }
                    default:
                        return fail("invalid escape character");
                }
            } else if (static_cast<unsigned char>(c) < 0x20) {
                return fail("control character in string");
            } else {
                out->push_back(c);
            }
        }
    }

    bool parseNumber(Value* out) {
        const size_t start = pos_;
        if (peek() == '-') {
            ++pos_;
        }
        if (pos_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(peek()))) {
            return fail("invalid number");
        }
        while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
            ++pos_;
        }
        bool isReal = false;
        if (peek() == '.') {
            isReal = true;
            ++pos_;
            if (pos_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
                return fail("invalid fraction");
            }
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
                ++pos_;
            }
        }
        if (peek() == 'e' || peek() == 'E') {
            isReal = true;
            ++pos_;
            if (peek() == '+' || peek() == '-') {
                ++pos_;
            }
            if (pos_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
                return fail("invalid exponent");
            }
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
                ++pos_;
            }
        }
        const std::string token = text_.substr(start, pos_ - start);
        if (!isReal) {
            errno = 0;
            char* end = nullptr;
            const long long value = std::strtoll(token.c_str(), &end, 10);
            if (errno == 0 && end != nullptr && *end == '\0') {
                *out = Value::int_(value);
                return true;
            }
            // Integer literal beyond int64: fall through to double like Python json.
        }
        const double value = std::strtod(token.c_str(), nullptr);
        if (!std::isfinite(value)) {
            return fail("JSON numbers must be finite");
        }
        *out = Value::real_(value);
        return true;
    }

    const std::string& text_;
    size_t pos_ = 0;
    std::string error_;
};

} // namespace

bool parse(const std::string& text, Value* out, std::string* error) {
    Parser parser(text);
    if (!parser.parse(out)) {
        if (error != nullptr) {
            *error = parser.error();
        }
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// writer
// ---------------------------------------------------------------------------

std::string escapeString(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 8);
    for (char c : input) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
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

namespace {

void writeValue(const Value& value, std::string* out) {
    char buf[64];
    switch (value.kind) {
        case Value::NUL:
            *out += "null";
            break;
        case Value::BOOL:
            *out += value.boolean ? "true" : "false";
            break;
        case Value::INT:
            std::snprintf(buf, sizeof(buf), "%lld", value.integer);
            *out += buf;
            break;
        case Value::REAL:
            // Full round-trip precision; probabilities are read back by ArkTS.
            std::snprintf(buf, sizeof(buf), "%.17g", value.real);
            *out += buf;
            break;
        case Value::STR:
            out->push_back('"');
            *out += escapeString(value.str);
            out->push_back('"');
            break;
        case Value::ARR:
            out->push_back('[');
            for (size_t i = 0; i < value.arr.size(); ++i) {
                if (i > 0) {
                    out->push_back(',');
                }
                writeValue(value.arr[i], out);
            }
            out->push_back(']');
            break;
        case Value::OBJ:
            out->push_back('{');
            for (size_t i = 0; i < value.obj.size(); ++i) {
                if (i > 0) {
                    out->push_back(',');
                }
                out->push_back('"');
                *out += escapeString(value.obj[i].first);
                *out += "\":";
                writeValue(value.obj[i].second, out);
            }
            out->push_back('}');
            break;
    }
}

} // namespace

std::string write(const Value& value) {
    std::string out;
    writeValue(value, &out);
    return out;
}

} // namespace djson
} // namespace pocket