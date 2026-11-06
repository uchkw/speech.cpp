#include "words.h"

#include <algorithm>

#include "guard.h"
#include "recognizer.h"
#include "unicode.h"
#include "utf8.h"

namespace fastconformer {

namespace {

/** SentencePiece's word-start piece, U+2581. */
const std::string kWordStart = "\xE2\x96\x81";

/**
 * SentencePiece's model of the file's tokenizer. Its pieces are normal but for the unknown piece and the byte pieces:
 * the converter writes a tokenizer of the kind haas-asr only where it has no other.
 */
PieceModel piece_model(const ModelFile & m) {
    const std::string p = "fastconformer.tokenizer.";
    PieceModel model;
    model.pieces = m.str_array(p + "tokens");
    model.scores = m.f32_array(p + "scores");
    model.types.assign(model.pieces.size(), PieceType::Normal);
    model.types[m.u32(p + "unknown_id")] = PieceType::Unknown;
    for (int32_t id : m.i32_array(p + "byte_pieces")) model.types[(size_t) id] = PieceType::Byte;
    const std::vector<uint8_t> charsmap = m.u8_array(p + "charsmap");
    model.charsmap.assign(charsmap.begin(), charsmap.end());
    model.add_dummy_prefix = m.boolean(p + "add_dummy_prefix");
    model.remove_extra_whitespaces = m.boolean(p + "remove_extra_whitespaces");
    model.escape_whitespaces = m.boolean(p + "escape_whitespaces");
    return model;
}

/** `text` without the whitespace str.strip() removes at either end. */
std::string strip(const std::string & text) {
    const std::vector<uint32_t> cps = *decode_utf8(text);
    size_t a = 0, b = cps.size();
    while (a < b && python_space(cps[a])) a++;
    while (b > a && python_space(cps[b - 1])) b--;
    return encode_utf8(std::vector<uint32_t>(cps.begin() + (std::ptrdiff_t) a, cps.begin() + (std::ptrdiff_t) b));
}

}  // namespace

std::vector<std::string> word_entries(const std::string & words) {
    std::vector<std::string> entries;
    size_t from = 0;
    while (from <= words.size()) {
        const size_t end = std::min(words.find('\n', from), words.size());
        const std::string line = words.substr(from, end - from);
        if (!strip(line).empty()) entries.push_back(line);
        from = end + 1;
    }
    return entries;
}

PreparedWords::PreparedWords(const std::string & raw, const std::vector<std::string> & entries, float alpha)
    : raw(raw), entry_count(entries.size()), vocabulary(entries, alpha) {}

size_t PreparedWords::tree_bytes() const {
    return boosting ? boosting->arcs().capacity() * sizeof(BoostingTree::Arc) +
                          boosting->states().capacity() * sizeof(BoostingTree::State) : 0;
}

PreparedWords & WordsCache::prepare(const std::string & raw, const std::vector<std::string> & entries, float alpha,
                                   std::unique_ptr<PreparedWords> & pending) {
    pending.reset();
    if (enabled_ && last_ && last_->raw == raw && last_->vocabulary.alpha() == alpha) return *last_;
    pending = std::make_unique<PreparedWords>(raw, entries, alpha);
    return *pending;
}

bool WordsCache::publish(std::unique_ptr<PreparedWords> & pending) {
    if (!enabled_ || !pending || !pending->boosting || pending->raw.size() > kRawBytes ||
        pending->entry_count > kEntries || pending->tree_bytes() > kTreeBytes) return false;
    last_ = std::move(pending);
    return true;
}

bool WordsModel::offers(const ModelFile & m) {
    return m.one_of("fastconformer.vocabulary.kind", {"none", "haas-asr"}) == "haas-asr";
}

WordsModel::WordsModel(const ModelFile & m)
    : encoder_(piece_model(m)),
      pieces_(m.str_array("fastconformer.tokenizer.tokens")),
      context_score_(m.f32("fastconformer.vocabulary.context_score")),
      depth_scaling_(m.f32("fastconformer.vocabulary.depth_scaling")),
      unk_score_(m.f32("fastconformer.vocabulary.unk_score")),
      alpha_(m.f32("fastconformer.vocabulary.alpha")),
      speech_rms_(m.f32("fastconformer.vocabulary.speech_rms")),
      speech_frame_(m.size("fastconformer.vocabulary.speech_frame")) {
    for (size_t i = 0; i < pieces_.size(); i++) id_of_.emplace(pieces_[i], (int32_t) i);
}

std::vector<int32_t> WordsModel::ids_of(const std::vector<std::string> & pieces, bool & found) const {
    std::vector<int32_t> ids;
    found = true;
    for (const std::string & piece : pieces) {
        const auto it = id_of_.find(piece);
        if (it == id_of_.end()) {
            found = false;
            return {};
        }
        ids.push_back(it->second);
    }
    return ids;
}

std::vector<std::vector<int32_t>> WordsModel::word_forms(const std::string & spelling) const {
    const std::vector<int32_t> ids = encoder_.encode(spelling);
    const std::string first = ids.empty() ? "" : pieces_[(size_t) ids[0]];
    if (first.compare(0, kWordStart.size(), kWordStart) != 0) return {ids};
    const std::string rest = first.substr(kWordStart.size());
    // `_pieces([rest]) or _pieces(list(rest)) if rest else []`: the merged piece's text as one piece, or else one a
    // character, or nothing where the word-start piece stands alone.
    bool found = true;
    std::vector<int32_t> inside;
    if (!rest.empty()) {
        inside = ids_of({rest}, found);
        if (!found) {
            const std::vector<uint32_t> code_points = *decode_utf8(rest);
            std::vector<std::string> characters;
            for (uint32_t c : code_points) characters.push_back(encode_utf8({c}));
            inside = ids_of(characters, found);
        }
    }
    if (!found || (inside.empty() && ids.size() == 1)) return {ids};
    inside.insert(inside.end(), ids.begin() + 1, ids.end());
    return {inside, ids};
}

BoostingTree WordsModel::tree(const Vocabulary & vocabulary) const {
    std::vector<std::vector<int32_t>> words;
    for (const std::string & spelling : vocabulary.spellings()) {
        for (std::vector<int32_t> & form : word_forms(spelling)) words.push_back(std::move(form));
    }
    return BoostingTree(words, (int32_t) pieces_.size(), context_score_, depth_scaling_, unk_score_);
}


WordsText WordsModel::recognize(Recognizer & recognizer, const Vocabulary & vocabulary, bool guard, const std::vector<float> & samples,
                                const std::vector<float> & projected, const std::string & name, DecodeOn on,
                                const DecodingProgress & progress) const {
    std::optional<BoostingTree> boosting;
    return recognize(recognizer, vocabulary, boosting, guard, samples, projected, name, on, progress);
}

WordsText WordsModel::recognize(Recognizer & recognizer, PreparedWords & prepared, bool guard, const std::vector<float> & samples,
                                const std::vector<float> & projected, const std::string & name, DecodeOn on,
                                const DecodingProgress & progress) const {
    return recognize(recognizer, prepared.vocabulary, prepared.boosting, guard, samples, projected, name, on, progress);
}

WordsText WordsModel::recognize(Recognizer & recognizer, const Vocabulary & vocabulary, std::optional<BoostingTree> & boosting,
                                bool guard, const std::vector<float> & samples, const std::vector<float> & projected,
                                const std::string & name, DecodeOn on, const DecodingProgress & progress) const {
    const Detokenizer & detokenizer = recognizer.detokenizer();
    WordsText out;
    const Decoding plain = recognizer.decoding(projected, name, on, [&](double done) { return !progress || progress(done / 2); });
    out.plain = strip(detokenizer.text(plain.ids));
    out.speech = has_speech(samples, speech_rms_, (size_t) speech_frame_);
    std::string text = out.plain;
    if (out.speech) {
        if (!boosting) boosting.emplace(tree(vocabulary));
        const Decoding boosted = recognizer.decoding(projected, name, on, [&](double done) { return !progress || progress((1 + done) / 2); },
                                                     Boost{*boosting, vocabulary.alpha()});
        out.boosted = strip(detokenizer.text(boosted.ids));
        if (out.boosted != out.plain) text = guard ? guarded(out.plain, out.boosted, vocabulary) : out.boosted;
    }
    out.text = vocabulary.format(text);
    return out;
}

}  // namespace fastconformer
