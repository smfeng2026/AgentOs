/*
 * Minimal JSON value tree used by the decision pipeline.
 *
 * The decision protocol is JSON in / JSON out, and object insertion order is
 * semantically significant (question branches are encoded in request order),
 * so objects keep a vector of pairs instead of a hash map. Numbers stay split
 * into int / real because the training serializer renders them differently
 * (Python str(1) == "1" but str(1.0) == "1.0").
 */

#ifndef POCKET_DECISION_JSON_H
#define POCKET_DECISION_JSON_H

#include <string>
#include <utility>
#include <vector>

namespace pocket {
namespace djson {

struct Value {
    enum Kind { NUL, BOOL, INT, REAL, STR, ARR, OBJ };

    Kind kind = NUL;
    bool boolean = false;
    long long integer = 0;
    double real = 0.0;
    std::string str;
    std::vector<Value> arr;
    std::vector<std::pair<std::string, Value>> obj; // insertion order preserved

    bool isNull() const { return kind == NUL; }
    bool isObject() const { return kind == OBJ; }
    bool isArray() const { return kind == ARR; }
    bool isString() const { return kind == STR; }
    bool isScalar() const { return kind == BOOL || kind == INT || kind == REAL || kind == STR; }

    const Value* find(const std::string& key) const;
    Value* find(const std::string& key);

    static Value object() { Value v; v.kind = OBJ; return v; }
    static Value array() { Value v; v.kind = ARR; return v; }
    static Value string(const std::string& s) { Value v; v.kind = STR; v.str = s; return v; }
    static Value boolean_(bool b) { Value v; v.kind = BOOL; v.boolean = b; return v; }
    static Value int_(long long i) { Value v; v.kind = INT; v.integer = i; return v; }
    static Value real_(double d) { Value v; v.kind = REAL; v.real = d; return v; }

    void set(const std::string& key, Value value);
};

/** Strict recursive-descent parser. Rejects trailing garbage and non-finite numbers. */
bool parse(const std::string& text, Value* out, std::string* error);

/** Compact writer, UTF-8 passthrough (ensure_ascii=False), rejects non-finite reals. */
std::string write(const Value& value);

/** Escapes a string for embedding inside a JSON string literal (no surrounding quotes). */
std::string escapeString(const std::string& input);

} // namespace djson
} // namespace pocket

#endif // POCKET_DECISION_JSON_H