#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "transducer.h"

namespace fastconformer {

class Decoder;

/**
 * The prediction prefix of the last completed plain request of one KDA TDT decoder on one backend: at most
 * kEntries entries, each the prediction one fed label produced, in the order the labels were fed. A request reuses
 * an entry only while the whole fed-label prefix before it and the joint's choice at every earlier entry match its
 * own, so that a hit continues the same prediction history, which the prediction network takes from its labels
 * alone, while every joint runs on the request's own audio. An entry's next label is the label the joint chose
 * after that prediction, which a shorter request can have emitted without ever feeding: no entry of a step that
 * was not computed exists, and a longer request computes it. Haas's dimensions give each entry 46,720 state floats
 * and 640 projected prediction floats, 189,440 bytes, so that a generation of 64 takes 11.5625 MiB and twice that
 * while a candidate is under way (2026-11); the graph, the decoding's scratch and the allocator's storage are not
 * part of it, and another model's dimensions take another share.
 */
class TdtPrefixCache {
public:
    static constexpr size_t kEntries = 64;

    /** A prediction after its fed label, and the next label chosen by the joint on that request's audio. */
    struct Entry {
        int32_t label;
        PredictionState state;
        std::vector<float> prediction;
        int32_t next_label;
    };

    /** A new request's candidate, while the last completed request stays immutable until publication. */
    class Reading {
    public:
        Reading(Reading &&) = default;
        Reading & operator=(Reading &&) = delete;
        Reading(const Reading &) = delete;
        Reading & operator=(const Reading &) = delete;

        /**
         * Binds one fresh zero-state decoding to its immutable decoder and its actual backend: a candidate whose
         * cache holds another's, or holds none yet, matches nothing. The decoding must begin from the prediction
         * network's zero state and frame 0, which the decoder it is given does.
         */
        void bind(const Decoder * decoder, ggml_backend_t backend);
        /** The next entry where the whole fed-label prefix has matched so far, or null, which ends reuse for good. */
        const Entry * find(int32_t label);
        /** Records a prediction, and its joint's next label, which the next lookup is accepted against. */
        void append(int32_t label, const PredictionState & state, const std::vector<float> & prediction, int32_t next_label);
        /** Marks a decoding that reached its end without a progress callback cancelling it. */
        void finish() { complete_ = true; }

        bool complete() const { return complete_; }
        size_t hits() const { return hits_; }
        size_t size() const { return entries_.size(); }

    private:
        friend class TdtPrefixCache;
        Reading(const TdtPrefixCache & source, size_t samples);

        const TdtPrefixCache & source_;
        size_t samples_, step_ = 0, hits_ = 0;
        const Decoder * decoder_ = nullptr;
        ggml_backend_t backend_ = nullptr;
        bool matching_ = false, complete_ = false;
        std::vector<Entry> entries_;
    };

    /** Starts a candidate, dropping the published prefix where the request's audio is shorter. The model serializes requests. */
    Reading begin(size_t samples);
    /** Replaces the completed prefix, refusing a candidate that did not finish or belongs to another cache. */
    void publish(Reading && reading);
    /** Drops payload and identity after keywords, cancellation, errors or changed request options. */
    void clear() noexcept;

    /** The published entries, in the order their labels were fed; empty where nothing is published. */
    const std::vector<Entry> & entries() const { return entries_; }

private:
    size_t samples_ = 0;
    const Decoder * decoder_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    std::vector<Entry> entries_;
};

}  // namespace fastconformer
