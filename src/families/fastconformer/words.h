#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "boosting.h"
#include "decoder.h"
#include "model-file.h"
#include "sentencepiece.h"
#include "vocabulary.h"

namespace fastconformer {

class Recognizer;
enum class DecodeOn;

/** The entries of words given one a line, as Vocabulary.parse() takes each: the lines, but for those of whitespace alone. */
std::vector<std::string> word_entries(const std::string & words);

/** What Haas-ASR's recognition with words found (haas_asr.boosting.transcribe()). */
struct WordsText {
    /** The texts of the decodings without and with the words, stripped; the second empty where `speech` is false. */
    std::string plain, boosted;
    /** Whether the audio has speech, has_speech(): without it the words are not boosted. */
    bool speech;
    /** The text: the plain one, the guarded or the boosted one, with the words in their written forms. */
    std::string text;
};

/**
 * What a file of fastconformer.vocabulary.kind haas-asr holds for Haas-ASR's custom vocabulary: the tokenizer's
 * encoding, the tree's settings, the default alpha and has_speech()'s threshold and frame.
 */
class WordsModel {
public:
    /** The settings of the file `m`, which offers the vocabulary (offers()). */
    explicit WordsModel(const ModelFile & m);

    /** Whether the file `m` offers Haas-ASR's custom vocabulary. */
    static bool offers(const ModelFile & m);

    /**
     * The token sequences a spelling is boosted as, haas_asr.boosting.word_forms(): SentencePiece's encoding, which
     * begins with the word-start piece a Japanese transcript has only where an utterance begins, and, where that piece
     * is one of its own or merged into the first, the same tokens without it, after the encoding, the merged piece's text
     * as one piece or one a character where the vocabulary has them.
     */
    std::vector<std::vector<int32_t>> word_forms(const std::string & spelling) const;

    /** The tree of every spelling of `vocabulary` in each of its word forms, in order, as boosting._compiled() builds it. */
    BoostingTree tree(const Vocabulary & vocabulary) const;

    /**
     * Haas-ASR's text of `samples` with `vocabulary`'s words, from `projected`, their encoder's output, by the decoding
     * `name` computed where `on` says: the plain decoding's text where the audio has no speech, and otherwise, decoded
     * again with the words boosted at `vocabulary`'s alpha, the boosted text where it differs, guarded where `guard` is
     * set, with the words in their written forms. `progress` hears the two decodings in turn, as halves.
     */
    WordsText recognize(Recognizer & recognizer, const Vocabulary & vocabulary, bool guard, const std::vector<float> & samples,
                        const std::vector<float> & projected, const std::string & name, DecodeOn on, const DecodingProgress & progress) const;

    float alpha() const { return alpha_; }
    float speech_rms() const { return speech_rms_; }
    int speech_frame() const { return speech_frame_; }

private:
    /** The ids of `pieces` where the vocabulary holds every one, boosting._pieces(); empty with `found` false otherwise. */
    std::vector<int32_t> ids_of(const std::vector<std::string> & pieces, bool & found) const;

    PieceEncoder encoder_;
    std::vector<std::string> pieces_;
    std::map<std::string, int32_t> id_of_;
    double context_score_, depth_scaling_;
    float unk_score_, alpha_, speech_rms_;
    int speech_frame_;
    bool keyword_prefix_;
};

}  // namespace fastconformer
