// Checks last-one keyword ownership, construction/publication counts and admission without a model or GPU.
// The tree uses deterministic synthetic token forms; it does not certify SentencePiece or official model outputs.
// Usage: fastconformer-keyword-cache-check

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "error.h"
#include "keyword-cache-check.h"

using namespace fastconformer;
using namespace keyword_check;

namespace {

BoostingTree synthetic_tree(const Vocabulary & vocabulary) {
    std::vector<std::vector<int32_t>> forms;
    for (const auto & spelling : vocabulary.spellings()) {
        std::vector<int32_t> ids;
        for (unsigned char byte : spelling) ids.push_back(1 + byte % 64);
        forms.push_back(std::move(ids));
    }
    return BoostingTree(forms, 65, 1.0, 1.0, 0.0f);
}

void build_tree(PreparedWords & prepared, Counters & counts) {
    if (!prepared.boosting) {
        prepared.boosting.emplace(synthetic_tree(prepared.vocabulary));
        ++counts.trees;
    }
}

void content(PreparedWords & prepared, const std::string & raw, float alpha, Counters & counts, Checks & check) {
    const Vocabulary reference(word_entries(raw), alpha);
    vocabulary(prepared.vocabulary, reference, check);
    build_tree(prepared, counts);
    tree(*prepared.boosting, synthetic_tree(reference), check);
}

std::string entries(size_t count) {
    std::string text;
    while (count--) text += "Haas\n";
    return text;
}

}  // namespace

int main() {
    try {
        Checks check;
        Counters counts;
        WordsCache cache(true);
        const std::string a = "Haas=ハース,はーす\nAIモデル", b = "ABC=えーびーしー\nabc=エービーシー";
        PreparedWords * first = nullptr;
        for (int request = 0; request < 3; ++request) {
            std::unique_ptr<PreparedWords> pending;
            auto & prepared = prepare(cache, a, 2.0f, pending, counts);
            if (request == 0) first = &prepared;
            check.expect(&prepared == first && bool(pending) == (request == 0), "Repeated A did not reuse its actual preparation.");
            content(prepared, a, 2.0f, counts, check);
            check.expect(publish(cache, pending, counts) == (request == 0), "Repeated A published another entry.");
        }
        check.expect(counts.preparations == 1 && counts.trees == 1 && counts.publications == 1, "Repeated A rebuilt its preparation.");
        for (const auto & raw : {b, a}) {
            std::unique_ptr<PreparedWords> pending;
            auto & prepared = prepare(cache, raw, 2.0f, pending, counts);
            check.expect(bool(pending), "A/B/A must miss after eviction.");
            content(prepared, raw, 2.0f, counts, check);
            check.expect(publish(cache, pending, counts), "A completed small entry was not retained.");
        }
        {
            std::unique_ptr<PreparedWords> pending;
            auto & same = prepare(cache, a, (float) std::nextafter(2.0, 3.0), pending, counts);
            check.expect(!pending && same.vocabulary.alpha() == 2.0f, "The key did not use the effective float alpha.");
            auto & changed = prepare(cache, a, 2.5f, pending, counts);
            check.expect(bool(pending), "Changing the effective alpha must miss.");
            content(changed, a, 2.5f, counts, check);
            publish(cache, pending, counts);
        }
        const std::vector<std::string> ordered = {
            "Haas=ハース\nHaas=はーす", "Haas=はーす\nHaas=ハース",
            "Cafe\xCC\x81=カフェ\nCafé=かふぇ", "Café=かふぇ\nCafe\xCC\x81=カフェ",
            " ab=えーびー \na=えー\n\n", "a=えー\nab=えーびー",
        };
        for (const auto & raw : ordered) {
            std::unique_ptr<PreparedWords> pending;
            auto & prepared = prepare(cache, raw, 2.0f, pending, counts);
            check.expect(bool(pending), "A distinct raw ordered list must miss.");
            content(prepared, raw, 2.0f, counts, check);
            publish(cache, pending, counts);
        }
        {
            std::unique_ptr<PreparedWords> pending;
            auto & prepared = prepare(cache, a, 2.0f, pending, counts);
            content(prepared, a, 2.0f, counts, check);
            publish(cache, pending, counts);
        }
        for (bool complete : {false, true}) {
            std::unique_ptr<PreparedWords> abandoned;
            auto & prepared = prepare(cache, b, 2.0f, abandoned, counts);
            if (complete) build_tree(prepared, counts);
            else check.expect(!publish(cache, abandoned, counts), "A treeless preparation must not be retained.");
        }
        bool refused = false;
        try {
            std::unique_ptr<PreparedWords> invalid;
            prepare(cache, "=はーす", 2.0f, invalid, counts);
        } catch (const Error & error) { refused = error.fault() == Fault::InvalidArgument; }
        check.expect(refused, "An invalid vocabulary was accepted.");
        {
            std::unique_ptr<PreparedWords> pending;
            auto & prepared = prepare(cache, a, 2.0f, pending, counts);
            check.expect(!pending, "An invalid or abandoned request evicted the previous valid entry.");
            content(prepared, a, 2.0f, counts, check);
            WordsCache independent(true);
            std::unique_ptr<PreparedWords> other;
            auto & separate = prepare(independent, a, 2.0f, other, counts);
            check.expect(other && &separate != &prepared, "Different owners shared a preparation.");
        }
        for (const auto & test : std::vector<std::pair<std::string, bool>>{
                 {std::string(WordsCache::kRawBytes, 'a'), true}, {std::string(WordsCache::kRawBytes + 1, 'a'), false},
                 {entries(WordsCache::kEntries), true}, {entries(WordsCache::kEntries + 1), false}}) {
            WordsCache bounded(true);
            std::unique_ptr<PreparedWords> pending;
            auto & prepared = prepare(bounded, test.first, 2.0f, pending, counts);
            build_tree(prepared, counts);
            check.expect(publish(bounded, pending, counts) == test.second, "A raw-byte or entry-count admission boundary differs.");
        }
        {
            std::unique_ptr<PreparedWords> pending;
            auto & prepared = prepare(cache, b, 2.0f, pending, counts);
            prepared.boosting.emplace(std::vector<std::vector<int32_t>>{},
                                     (int32_t) (WordsCache::kTreeBytes / sizeof(BoostingTree::Arc) + 1), 1.0, 1.0, 0.0f);
            ++counts.trees;
            check.expect(prepared.tree_bytes() > WordsCache::kTreeBytes && !publish(cache, pending, counts),
                         "An over-limit allocated tree payload was retained.");
            auto & previous = prepare(cache, a, 2.0f, pending, counts);
            check.expect(!pending && previous.vocabulary.alpha() == 2.0f, "An over-limit request evicted the previous entry.");
        }
        WordsCache disabled(false);
        Counters off;
        for (int request = 0; request < 2; ++request) {
            std::unique_ptr<PreparedWords> pending;
            auto & prepared = prepare(disabled, a, 2.0f, pending, off);
            build_tree(prepared, off);
            check.expect(!publish(disabled, pending, off) && pending, "The disabled cache retained an entry.");
        }
        check.expect(off.preparations == 2 && off.trees == 2 && off.publications == 0, "The disabled path reused preparation.");
        std::printf("keyword cache: %zu checks; preparations=%zu trees=%zu publications=%zu; no model/GPU/official reference/timing\n",
                    check.total, counts.preparations, counts.trees, counts.publications);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
