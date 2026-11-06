#pragma once

#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "fastconformer/words.h"

namespace keyword_check {

struct Checks {
    size_t total = 0;
    void expect(bool condition, const std::string & what) {
        ++total;
        if (!condition) throw std::runtime_error(what);
    }
};

/** Successful constructions and ownership transfers observed by the check, not stored in the runtime. */
struct Counters {
    size_t preparations = 0, trees = 0, publications = 0;
};

inline fastconformer::PreparedWords & prepare(fastconformer::WordsCache & cache, const std::string & raw, float alpha,
                                             std::unique_ptr<fastconformer::PreparedWords> & pending, Counters & counts) {
    auto & prepared = cache.prepare(raw, fastconformer::word_entries(raw), alpha, pending);
    if (pending) ++counts.preparations;
    return prepared;
}

inline bool publish(fastconformer::WordsCache & cache, std::unique_ptr<fastconformer::PreparedWords> & pending, Counters & counts) {
    const bool kept = cache.publish(pending);
    if (kept) ++counts.publications;
    return kept;
}

template <typename T>
bool bits(const std::vector<T> & a, const std::vector<T> & b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

inline bool same_float(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }

inline void vocabulary(const fastconformer::Vocabulary & got, const fastconformer::Vocabulary & want, Checks & check) {
    check.expect(same_float(got.alpha(), want.alpha()), "The effective alpha differs.");
    check.expect(got.spellings() == want.spellings() && got.words().size() == want.words().size(), "The vocabulary shape differs.");
    for (size_t i = 0; i < got.words().size(); ++i) {
        check.expect(got.words()[i].text == want.words()[i].text && got.words()[i].readings == want.words()[i].readings,
                     "The ordered written forms or readings differ.");
    }
    for (const std::string text : {"Haas とハース，はーす AIモデル", "Cafe\xCC\x81 CAFÉ カフェ", "a ab abc ABC", ""}) {
        check.expect(got.format(text) == want.format(text), "The formatting forms differ.");
    }
}

inline void tree(const fastconformer::BoostingTree & got, const fastconformer::BoostingTree & want, Checks & check) {
    check.expect(got.vocab_size() == want.vocab_size() && got.arcs().size() == want.arcs().size() &&
                 got.states().size() == want.states().size(), "The tree shape differs.");
    for (size_t i = 0; i < got.arcs().size(); ++i) {
        const auto & a = got.arcs()[i];
        const auto & b = want.arcs()[i];
        check.expect(a.from == b.from && a.to == b.to && a.token == b.token && same_float(a.weight, b.weight), "An arc differs.");
    }
    for (size_t i = 0; i < got.states().size(); ++i) {
        const auto & a = got.states()[i];
        const auto & b = want.states()[i];
        check.expect(a.arcs_start == b.arcs_start && a.arcs_end == b.arcs_end && a.order == b.order &&
                     a.backoff_to == b.backoff_to && same_float(a.backoff_weight, b.backoff_weight), "A tree state differs.");
        std::vector<float> left, right;
        std::vector<int32_t> next_left, next_right;
        got.advance((int32_t) i, left, next_left);
        want.advance((int32_t) i, right, next_right);
        check.expect(bits(left, right) && next_left == next_right, "A state's score or next-state vector differs.");
    }
}

inline void words_text(const fastconformer::WordsText & got, const fastconformer::WordsText & want, Checks & check) {
    check.expect(got.plain == want.plain && got.boosted == want.boosted && got.speech == want.speech && got.text == want.text,
                 "The plain, boosted, speech decision or formatted text differs.");
}

inline void decoding(const fastconformer::Decoding & got, const fastconformer::Decoding & want, Checks & check) {
    check.expect(got.ids == want.ids && got.frames == want.frames && got.durations == want.durations && got.graphs == want.graphs,
                 "Raw token IDs, frames, durations or graph count differ.");
}

}  // namespace keyword_check
