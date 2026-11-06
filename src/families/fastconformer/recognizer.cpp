#include "recognizer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "backend.h"
#include "error.h"
#include "layout.h"

namespace fastconformer {

namespace {

bool experiment_requested(const char * name) {
    const char * value = std::getenv(name);
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw Error(Fault::InvalidArgument, std::string(name) + " must be 0 or 1");
}

}  // namespace

struct Recognizer::EncoderGraph {
    Graph graph;
    ggml_tensor * input;
    ggml_tensor * output;
    int64_t frames;

    EncoderGraph(const Encoder & encoder, const Joint & joint, const std::vector<float> & features, int mels, int64_t frames)
        : frames(frames) {
        input = graph.input(features, mels, frames);
        output = joint.project_encoder(graph.ctx(), encoder.build(graph, input));
        graph.output(output);
    }
};

Recognizer::Transducer::Transducer(const ModelFile & m, const std::vector<std::string> & names) : prediction(m), joint(m), names(names) {
    for (const std::string & name : names) decoders.push_back(make_decoder(m, prediction, joint, name));
}

const Decoder & Recognizer::Transducer::decoder(const std::string & name) const {
    for (size_t i = 0; i < names.size(); i++) {
        if (names[i] == name) return *decoders[i];
    }
    throw std::logic_error("a FastConformer model has no decoding named " + name);
}

Recognizer::OnCpu::OnCpu(const ModelFile & device, int threads, const std::vector<std::string> & names)
    : pool(nullptr, ggml_threadpool_free),
      backend(start_device(find_device("cpu"), threads), ggml_backend_free),
      model(device, backend.get(), decoding_tensor),
      transducer(model, names) {
    ggml_threadpool_params params = ggml_threadpool_params_default(threads);
    pool.reset(ggml_threadpool_new(&params));
    if (!pool) throw Error(Fault::OutOfMemory, "cannot start the CPU's threads for the decoding");
    ggml_backend_cpu_set_threadpool(backend.get(), pool.get());
}

Recognizer::Recognizer(const std::string & path, ggml_backend_t backend, int threads)
    : backend_(backend),
      on_cpu_(ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU),
      threads_(threads),
      model_(path, backend, layout),
      frontend_(model_, "fastconformer.", true),
      encoder_(model_, backend),
      decoding_names_(fastconformer::decodings(model_)),
      device_(model_, decoding_names_),
      detokenizer_(model_),
      separators_(model_.str_array("fastconformer.segment.separators")),
      breaks_(model_.str_array("fastconformer.segment.breaks")),
      cache_encoder_graph_(experiment_requested("SPEECH_HAAS_ENCODER_GRAPH_CACHE") && model_.str("general.name") == "Haas-ja-v1.0" &&
                           model_.str("fastconformer.decoder.prediction") == "kda"),
      trace_encoder_graph_cache_(experiment_requested("SPEECH_HAAS_ENCODER_GRAPH_CACHE_TRACE")),
      allocr_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))) {
    if (!allocr_) throw Error(Fault::OutOfMemory, "cannot create a graph allocator");
    if (WordsModel::offers(model_)) words_ = std::make_unique<WordsModel>(model_);
}

Recognizer::~Recognizer() {
    if (trace_encoder_graph_cache_) {
        const auto info = encoder_graph_cache_info();
        std::fprintf(stderr, "haas_encoder_graph_cache enabled=%d builds=%zu hits=%zu frames=%lld feature_bytes=%zu compute_bytes=%zu\n",
                     (int) info.enabled, info.builds, info.hits, (long long) info.frames, info.feature_bytes, info.compute_bytes);
    }
    encoder_graph_.reset();
    ggml_gallocr_free(allocr_);
}

EncoderGraphCacheInfo Recognizer::encoder_graph_cache_info() const {
    return {cache_encoder_graph_, encoder_graph_builds_, encoder_graph_hits_, encoder_graph_ ? encoder_graph_->frames : -1,
            encoder_graph_ ? ggml_nbytes(encoder_graph_->input) : 0,
            ggml_gallocr_get_buffer_size(allocr_, 0)};
}

std::vector<float> Recognizer::encode(const std::vector<float> & features, int64_t frames) {
    if (cache_encoder_graph_) {
        const bool hit = encoder_graph_ && encoder_graph_->frames == frames;
        try {
            if (hit) {
                encoder_graph_->graph.set(encoder_graph_->input, features);
                encoder_graph_->graph.compute_again(backend_);
            } else {
                encoder_graph_.reset();
                encoder_graph_ = std::make_unique<EncoderGraph>(encoder_, device_.joint, features, frontend_.mels(), frames);
                encoder_graph_->graph.compute(backend_, allocr_);
            }
            auto result = Graph::read(encoder_graph_->output);
            if (hit) ++encoder_graph_hits_;
            else ++encoder_graph_builds_;
            return result;
        } catch (...) {
            encoder_graph_.reset();
            throw;
        }
    }
    Graph g;
    ggml_tensor * out = device_.joint.project_encoder(g.ctx(), encoder_.build(g, features, frames));
    g.output(out);
    g.compute(backend_, allocr_);
    auto result = Graph::read(out);
    ++encoder_graph_builds_;
    return result;
}

Transcript Recognizer::recognize(const std::vector<float> & samples, const std::string & name, DecodeOn on) {
    const std::vector<float> features = frontend_.features(samples);
    Transcript t;
    t.decoding = decoding(encode(features, frontend_.frames(samples.size())), name, on);
    t.text = detokenizer_.text(t.decoding.ids);
    return t;
}

Recognizer::OnCpu & Recognizer::cpu() {
    if (!cpu_) cpu_ = std::make_unique<OnCpu>(model_, threads_, decoding_names_);
    return *cpu_;
}

Decoding Recognizer::decoding(const std::vector<float> & projected, const std::string & name, DecodeOn on,
                              const DecodingProgress & progress) {
    if (on == DecodeOn::Device || on_cpu_) return device_.decoder(name).decode(projected, backend_, progress);
    OnCpu & copy = cpu();
    return copy.transducer.decoder(name).decode(projected, copy.backend.get(), progress);
}

Decoding Recognizer::decoding(const std::vector<float> & projected, const std::string & name, DecodeOn on,
                              const DecodingProgress & progress, const Boost & boost) {
    if (on == DecodeOn::Device || on_cpu_) return device_.decoder(name).decode(projected, backend_, progress, boost);
    OnCpu & copy = cpu();
    return copy.transducer.decoder(name).decode(projected, copy.backend.get(), progress, boost);
}

std::vector<Segment> Recognizer::segments(const Decoding & decoding) const {
    return fastconformer::segments(detokenizer_.token_texts(decoding.ids), detokenizer_.word_starts(decoding.ids),
                                   token_spans(decoding, detokenizer_), separators_, breaks_);
}

double Recognizer::seconds(int64_t frame) const {
    const double window_stride = (double) frontend_.hop_length() / (double) frontend_.sample_rate();
    return (double) frame * window_stride * (double) encoder_.subsampling_factor();
}

}  // namespace fastconformer
