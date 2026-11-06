#pragma once

#include "json-reader.h"
#include "reference-dumps.h"

namespace fastconformer {

enum class DumpScope { Full, Encoder };

/** The stages a FastConformer dump's producer recorded, independent of which tensor files remain on disk. */
inline DumpScope dump_scope(const std::filesystem::path & dir) {
    const std::filesystem::path path = dir / "meta.json";
    const JsonValue meta = parse_json(dump_text(path));
    if (const JsonValue * scope = meta.member("reference_scope")) {
        if (scope->kind == JsonValue::Kind::String && scope->text == "frontend-encoder-joint.enc") return DumpScope::Encoder;
        throw std::runtime_error(path.u8string() + ": unknown FastConformer reference_scope");
    }
    // dump.py records these three fields for every complete dump; dump-encoder.py records its explicit scope instead.
    const JsonValue * prediction = meta.member("prediction_steps"), * joint = meta.member("joint_evaluations"), * text = meta.member("text");
    if (prediction && prediction->is_integer() && joint && joint->is_integer() && text && text->kind == JsonValue::Kind::String) {
        return DumpScope::Full;
    }
    throw std::runtime_error(path.u8string() + ": no recognized FastConformer dump producer metadata");
}

/** Dumps of the requested scope; missing stage inputs within that scope still fail in the stage's reader. */
inline std::vector<std::filesystem::path> scoped_dumps(const std::string & root, const ModelFile & model, DumpScope scope) {
    std::vector<std::filesystem::path> selected;
    for (const auto & dir : reference_dumps(root, model)) {
        if (dump_scope(dir) == scope) selected.push_back(dir);
    }
    if (selected.empty()) {
        const std::string needed = scope == DumpScope::Full ? "full" : "frontend-encoder-joint.enc";
        const std::string help = scope == DumpScope::Full ? "; encoder-only dumps contain no prediction or decoding reference"
                                                         : "; create projection references with dump-encoder.py";
        throw std::runtime_error("no compatible " + needed + " FastConformer dumps under " + root + help);
    }
    return selected;
}

}  // namespace fastconformer
