#include "model-file.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <set>
#include <stdexcept>

#include "error.h"
#include "json.h"
#include "log.h"

namespace {

using Gguf = std::unique_ptr<gguf_context, decltype(&gguf_free)>;

/**
 * The bytes of a tensor read from the file and set on the device at a time. A buffer that held a whole tensor raised the
 * peak memory footprint by the largest one, Qwen3-TTS 1.7B's text embedding of 331 MB: by 315 MB with Metal and 321 MB
 * with the CPU on an Apple M5. ggml's Vulkan backend copies a write to a device's memory through a host buffer of its
 * own as large as the largest write, which it keeps until the process ends: a ready worker of Qwen3-ASR 1.7B held
 * 310 MB more private bytes on an RTX 2080. Pieces of 16 MB load the files as fast on both (2026-10-12).
 */
constexpr size_t kReadPiece = (size_t) 16 << 20;

Error file_error(const std::string & message) {
    return Error(Fault::File, message);
}

/**
 * The file's metadata. gguf_init_from_file() answers a file it cannot open and one that is not GGUF alike, so the
 * file is opened first to tell the two apart.
 */
Gguf open_gguf(const std::string & path, ggml_context ** ctx) {
    route_ggml_log();
    std::unique_ptr<FILE, decltype(&std::fclose)> f(ggml_fopen(path.c_str(), "rb"), &std::fclose);
    if (!f) throw Error(Fault::Io, "cannot open " + path + "; check the path and that the file can be read");
    f.reset();
    gguf_init_params params = {/*no_alloc =*/true, /*ctx =*/ctx};
    Gguf gguf(gguf_init_from_file(path.c_str(), params), gguf_free);
    if (!gguf) throw file_error("cannot read " + path + " as a GGUF file; check that it is a model file and that the download is complete");
    return gguf;
}

/**
 * The architectures of the files that the releases before layouts began wrote, which have no speech.layout. A file of
 * any other architecture without it was written for another program, such as audio.cpp or parakeet.cpp.
 */
const char * const kArchitecturesBeforeLayouts[] = {"dacvae", "fastconformer", "irodori-tts", "irodori-tts-voice", "qwen3tts-codec", "qwen3tts-talker"};

/**
 * Throws unless the file has speech.layout, which every file written for a release with layouts has, saying whether
 * the file was written for an earlier release of speech.cpp or for another program.
 */
void require_layout_key(const gguf_context * gguf, const std::string & path, const std::string & remedy) {
    if (gguf_find_key(gguf, "speech.layout") >= 0) return;
    const int64_t id = gguf_find_key(gguf, "general.architecture");
    const std::string architecture = id >= 0 && gguf_get_kv_type(gguf, id) == GGUF_TYPE_STRING ? gguf_get_val_str(gguf, id) : "";
    for (const char * before : kArchitecturesBeforeLayouts) {
        if (architecture == before) {
            throw file_error(path + " has no speech.layout: it was written for a release of speech.cpp before " + kFirstLayoutRelease +
                             ", which this release does not read; " + remedy);
        }
    }
    const std::string named = architecture.empty() ? " names no general.architecture" : " is a file of " + architecture;
    throw file_error(path + named + " and has no speech.layout: it was written for another program, and speech.cpp reads only its own " +
                     "files; use a model file for speech.cpp from its Hugging Face repository, or a voice file made with speech voice");
}

/** A key's type as the layouts write it: u32, or [i32] for an array. */
std::string type_text(gguf_type type, gguf_type element) {
    return type == GGUF_TYPE_ARRAY ? std::string("[") + gguf_type_name(element) + "]" : gguf_type_name(type);
}

std::string joined(const std::vector<std::string> & items, size_t at_most) {
    std::string out;
    for (size_t i = 0; i < items.size() && i < at_most; i++) out += (i ? ", " : "") + items[i];
    if (items.size() > at_most) out += " and " + std::to_string(items.size() - at_most) + " more";
    return out;
}

/** A shape as messages write it, "[2048, 1024]", without the axes of 1 past the last other one. */
std::string shape_text(const int64_t * ne) {
    int axes = GGML_MAX_DIMS;
    while (axes > 1 && ne[axes - 1] == 1) axes--;
    std::string out = "[";
    for (int i = 0; i < axes; i++) out += (i ? ", " : "") + std::to_string(ne[i]);
    return out + "]";
}

/** "F32", or "Q8_0, F16 or F32". */
std::string types_text(const std::vector<ggml_type> & types) {
    std::string out;
    for (size_t i = 0; i < types.size(); i++) out += (i == 0 ? "" : i + 1 == types.size() ? " or " : ", ") + tensor_type_text(types[i]);
    return out;
}

/** One value of type `type` at `data` as JSON. */
std::string scalar_json(gguf_type type, const void * data) {
    switch (type) {
        case GGUF_TYPE_UINT8: return std::to_string(*(const uint8_t *) data);
        case GGUF_TYPE_INT8: return std::to_string(*(const int8_t *) data);
        case GGUF_TYPE_UINT16: return std::to_string(*(const uint16_t *) data);
        case GGUF_TYPE_INT16: return std::to_string(*(const int16_t *) data);
        case GGUF_TYPE_UINT32: return std::to_string(*(const uint32_t *) data);
        case GGUF_TYPE_INT32: return std::to_string(*(const int32_t *) data);
        case GGUF_TYPE_UINT64: return std::to_string(*(const uint64_t *) data);
        case GGUF_TYPE_INT64: return std::to_string(*(const int64_t *) data);
        case GGUF_TYPE_FLOAT32: return json_number(*(const float *) data);
        case GGUF_TYPE_FLOAT64: return json_number(*(const double *) data);
        case GGUF_TYPE_BOOL: return *(const int8_t *) data ? "true" : "false";
        default: throw std::logic_error("scalar_json() was given a string or an array");
    }
}

}  // namespace

size_t gguf_scalar_size(gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
        default: throw std::logic_error("gguf_scalar_size() was given a string or an array");
    }
}

bool seek_file(FILE * f, uint64_t offset) {
#ifdef _WIN32
    return _fseeki64(f, (int64_t) offset, SEEK_SET) == 0;
#else
    return fseeko(f, (off_t) offset, SEEK_SET) == 0;
#endif
}

std::string gguf_architecture(const std::string & path) {
    const Gguf gguf = open_gguf(path, nullptr);
    require_layout_key(gguf.get(), path, "convert it again or download it again");
    const int64_t id = gguf_find_key(gguf.get(), "general.architecture");
    if (id < 0 || gguf_get_kv_type(gguf.get(), id) != GGUF_TYPE_STRING) {
        throw file_error(path + " names no general.architecture as a string");
    }
    return gguf_get_val_str(gguf.get(), id);
}

ModelFile::ModelFile(const std::string & path, ggml_backend_t backend, const Layout & layout,
                     const std::function<bool(const std::string & name)> & keep)
    : path_(path), architecture_(layout.architecture), remedy_(layout.remedy) {
    open(layout);
    allocate([&](const std::string & name) { return !keep || keep(name) ? backend : nullptr; });
    load();
}

ModelFile::ModelFile(const std::string & path, const Layout & layout, const std::function<ggml_backend_t(const std::string & name)> & place)
    : path_(path), architecture_(layout.architecture), remedy_(layout.remedy) {
    open(layout);
    allocate(place);
    load();
}

ModelFile::ModelFile(const std::string & path, const Layout & layout)
    : path_(path), architecture_(layout.architecture), remedy_(layout.remedy) {
    open(layout);
}

ModelFile::ModelFile(const ModelFile & source, ggml_backend_t backend, const std::function<bool(const std::string & name)> & keep,
                     const std::map<std::string, ggml_backend_buffer_type_t> & buffer_types)
    : path_(source.path_), architecture_(source.architecture_), remedy_(source.remedy_), version_(source.version_), gguf_(source.gguf_),
      ctx_(source.ctx_) {
    allocate([&](const std::string & name) { return (!keep || keep(name)) && source.by_name_.count(name) ? backend : nullptr; }, buffer_types);
    for (const auto & [name, t] : by_name_) ggml_backend_tensor_copy(source.tensor(name), t);
}

void ModelFile::open(const Layout & layout) {
    ggml_context * ctx = nullptr;
    gguf_ = open_gguf(path_, &ctx);
    ctx_.reset(ctx, ggml_free);
    require_layout_key(gguf_.get(), path_, remedy_);
    const int64_t id = gguf_find_key(gguf_.get(), "general.architecture");
    if (id < 0 || gguf_get_kv_type(gguf_.get(), id) != GGUF_TYPE_STRING) {
        throw file_error(path_ + " names no general.architecture as a string; " + remedy_);
    }
    const std::string architecture = gguf_get_val_str(gguf_.get(), id);
    if (architecture != layout.architecture) {
        throw file_error(path_ + " is a file of " + architecture + ", where a file of " + layout.architecture + " is expected");
    }
    version_ = u32("speech.layout");
    if (version_ > layout.version) {
        throw file_error(path_ + " has layout " + std::to_string(version_) + " of " + architecture_ + ", which speech.cpp " +
                         str("speech.requires") + " and later read; this is " + SPEECH_VERSION);
    }
    if (version_ != layout.version) {
        if (version_ == 0 || !layout.upgrade) {
            throw file_error(path_ + " has layout " + std::to_string(version_) + " of " + architecture_ + ", which no release of speech.cpp writes; " +
                             remedy_);
        }
        layout.upgrade(*this);
    }
    check_tensors(layout);
}

void ModelFile::check_tensors(const Layout & layout) const {
    const std::vector<TensorSpec> wanted = layout.tensors(*this);
    std::set<std::string> want;
    for (const TensorSpec & t : wanted) want.insert(t.name);
    std::set<std::string> have;
    std::vector<std::string> missing, extra;
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf_.get()); i++) {
        const std::string name = gguf_get_tensor_name(gguf_.get(), i);
        have.insert(name);
        if (!want.count(name)) extra.push_back(name);
    }
    for (const TensorSpec & t : wanted) {
        if (!have.count(t.name)) missing.push_back(t.name);
    }
    if (!missing.empty() || !extra.empty()) {
        std::string what;
        if (!missing.empty()) what += "it lacks " + joined(missing, 5);
        if (!extra.empty()) what += std::string(missing.empty() ? "" : ", and ") + "its keys call for no " + joined(extra, 5);
        throw file_error(path_ + " does not hold the tensors that " + layout_name() + " calls for: " + what + "; " + remedy_);
    }
    for (const TensorSpec & spec : wanted) {
        const ggml_tensor * t = ggml_get_tensor(ctx_.get(), spec.name.c_str());
        if (!std::equal(spec.shape.ne, spec.shape.ne + GGML_MAX_DIMS, t->ne)) {
            throw file_error("the tensor " + spec.name + " of " + path_ + " has the shape " + shape_text(t->ne) + ", where " + layout_name() +
                             " gives it the shape " + shape_text(spec.shape.ne) + "; " + remedy_);
        }
        const std::vector<ggml_type> types = spec.types();
        if (std::find(types.begin(), types.end(), t->type) == types.end()) {
            throw file_error("the tensor " + spec.name + " of " + path_ + " has the type " + tensor_type_text(t->type) + ", where " + layout_name() +
                             " stores it in " + types_text(types) + "; " + remedy_for_unknown());
        }
    }
}

void ModelFile::allocate(const std::function<ggml_backend_t(const std::string & name)> & place,
                         const std::map<std::string, ggml_backend_buffer_type_t> & buffer_types) {
    const int64_t n = gguf_get_n_tensors(gguf_.get());
    std::vector<std::pair<ggml_backend_t, ggml_backend_buffer_type_t>> places;
    for (int64_t i = 0; i < n; i++) {
        const char * name = gguf_get_tensor_name(gguf_.get(), i);
        ggml_backend_t backend = place(name);
        if (!backend) continue;
        const auto chosen = buffer_types.find(name);
        const auto placement = std::make_pair(backend, chosen == buffer_types.end() ? ggml_backend_get_default_buffer_type(backend) : chosen->second);
        const size_t at = (size_t) (std::find(places.begin(), places.end(), placement) - places.begin());
        if (at == places.size()) {
            ggml_init_params params = {ggml_tensor_overhead() * (size_t) n, nullptr, true};
            kept_.emplace_back(ggml_init(params), ggml_free);
            if (!kept_.back()) throw Error(Fault::OutOfMemory, "cannot create a ggml context for the tensors of " + path_);
            places.push_back(placement);
        }
        ggml_tensor * t = ggml_dup_tensor(kept_[at].get(), ggml_get_tensor(ctx_.get(), name));
        ggml_set_name(t, name);
        by_name_.emplace(name, t);
    }
    for (size_t i = 0; i < places.size(); i++) {
        buffers_.emplace_back(ggml_backend_alloc_ctx_tensors_from_buft(kept_[i].get(), places[i].second), ggml_backend_buffer_free);
        if (!buffers_.back()) throw Error(Fault::OutOfMemory, "cannot allocate the weights of " + path_ + " on " + ggml_backend_name(places[i].first));
        ggml_backend_buffer_set_usage(buffers_.back().get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
}

void ModelFile::load() {
    std::unique_ptr<FILE, decltype(&std::fclose)> f(ggml_fopen(path_.c_str(), "rb"), &std::fclose);
    if (!f) throw Error(Fault::Io, "cannot open " + path_);
    const uint64_t data_offset = gguf_get_data_offset(gguf_.get());
    std::vector<uint8_t> piece;
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf_.get()); i++) {
        const char * name = gguf_get_tensor_name(gguf_.get(), i);
        const auto it = by_name_.find(name);
        if (it == by_name_.end()) continue;
        ggml_tensor * t = it->second;
        const size_t size = ggml_nbytes(t);
        bool read = seek_file(f.get(), data_offset + gguf_get_tensor_offset(gguf_.get(), i));
        for (size_t done = 0; read && done < size;) {
            const size_t n = std::min(kReadPiece, size - done);
            piece.resize(std::max(piece.size(), n));
            read = std::fread(piece.data(), 1, n, f.get()) == n;
            if (read) ggml_backend_tensor_set(t, piece.data(), done, n);
            done += n;
        }
        if (!read) throw file_error(std::string("cannot read the tensor ") + name + " of " + path_ + "; the file is shorter than its header says");
    }
}

std::string ModelFile::layout_name() const {
    return version_ ? "layout " + std::to_string(version_) + " of " + architecture_ : architecture_;
}

std::string ModelFile::remedy_for_unknown() const {
    const int64_t id = gguf_find_key(gguf_.get(), "speech.requires");
    if (id >= 0 && gguf_get_kv_type(gguf_.get(), id) == GGUF_TYPE_STRING) {
        const std::string release = gguf_get_val_str(gguf_.get(), id);
        const auto numbers = release_numbers(release), own = release_numbers(SPEECH_VERSION);
        if (numbers && own && *numbers > *own) return "speech.cpp " + release + " and later read it, and this is " + SPEECH_VERSION;
    }
    return remedy_;
}

ggml_tensor * ModelFile::tensor(const std::string & name) const {
    const auto it = by_name_.find(name);
    if (it == by_name_.end()) throw std::logic_error("the tensor " + name + " of " + path_ + " is not loaded");
    return it->second;
}

bool ModelFile::has(const std::string & key) const {
    return gguf_find_key(gguf_.get(), key.c_str()) >= 0;
}

int64_t ModelFile::key_id(const std::string & key, gguf_type type, gguf_type element) const {
    const int64_t id = gguf_find_key(gguf_.get(), key.c_str());
    const std::string layout = layout_name();
    if (id < 0) throw file_error(path_ + " has no key " + key + ", which " + layout + " requires; " + remedy_);
    const gguf_type actual = gguf_get_kv_type(gguf_.get(), id);
    const gguf_type actual_element = actual == GGUF_TYPE_ARRAY ? gguf_get_arr_type(gguf_.get(), id) : GGUF_TYPE_COUNT;
    if (actual != type || actual_element != element) {
        throw file_error("the key " + key + " of " + path_ + " has the type " + type_text(actual, actual_element) + ", where " + layout +
                         " gives it the type " + type_text(type, element) + "; " + remedy_);
    }
    return id;
}

uint32_t ModelFile::u32(const std::string & key) const {
    return gguf_get_val_u32(gguf_.get(), key_id(key, GGUF_TYPE_UINT32));
}

void ModelFile::upgrade_u32(const std::string & key, uint32_t value) {
    gguf_set_val_u32(gguf_.get(), key.c_str(), value);
}

void ModelFile::upgrade_f32(const std::string & key, float value) {
    gguf_set_val_f32(gguf_.get(), key.c_str(), value);
}

void ModelFile::upgrade_bool(const std::string & key, bool value) {
    gguf_set_val_bool(gguf_.get(), key.c_str(), value);
}

void ModelFile::upgrade_str(const std::string & key, const std::string & value) {
    gguf_set_val_str(gguf_.get(), key.c_str(), value.c_str());
}

void ModelFile::upgrade_i32_array(const std::string & key, const std::vector<int32_t> & values) {
    gguf_set_arr_data(gguf_.get(), key.c_str(), GGUF_TYPE_INT32, values.data(), values.size());
}

void ModelFile::upgrade_f32_array(const std::string & key, const std::vector<float> & values) {
    gguf_set_arr_data(gguf_.get(), key.c_str(), GGUF_TYPE_FLOAT32, values.data(), values.size());
}

void ModelFile::upgrade_str_array(const std::string & key, const std::vector<std::string> & values) {
    std::vector<const char *> data;
    for (const std::string & v : values) data.push_back(v.c_str());
    gguf_set_arr_str(gguf_.get(), key.c_str(), data.data(), data.size());
}

int ModelFile::size(const std::string & key) const {
    const uint32_t value = u32(key);
    if (value == 0 || value > (uint32_t) INT_MAX) {
        throw file_error("the key " + key + " of " + path_ + " is " + std::to_string(value) + ", where " + layout_name() + " takes a size from 1 to " +
                         std::to_string(INT_MAX) + "; " + remedy_);
    }
    return (int) value;
}

int ModelFile::count(const std::string & key) const {
    const uint32_t value = u32(key);
    if (value == 0 || value > tensor_count()) {
        throw file_error("the key " + key + " of " + path_ + " is " + std::to_string(value) + ", where " + layout_name() +
                         " takes a number of layers or blocks from 1 to the " + std::to_string(tensor_count()) + " tensors the file holds; " + remedy_);
    }
    return (int) value;
}

int64_t ModelFile::width(const std::string & name, int axis) const {
    const ggml_tensor * t = ggml_get_tensor(ctx_.get(), name.c_str());
    if (!t) throw file_error(path_ + " does not hold the tensor " + name + ", which " + layout_name() + " calls for; " + remedy_);
    if (t->ne[axis] == 0 || t->ne[axis] > INT_MAX) {
        throw file_error("the tensor " + name + " of " + path_ + " has the shape " + shape_text(t->ne) + ", where " + layout_name() +
                         " takes a width from 1 to " + std::to_string(INT_MAX) + " on its axis " + std::to_string(axis) + "; " + remedy_);
    }
    return t->ne[axis];
}

ggml_type ModelFile::stored_type(const std::string & name) const {
    const ggml_tensor * t = ggml_get_tensor(ctx_.get(), name.c_str());
    if (!t) throw file_error(path_ + " does not hold the tensor " + name + ", which " + layout_name() + " calls for; " + remedy_);
    return t->type;
}

int64_t ModelFile::tensor_count() const {
    return gguf_get_n_tensors(gguf_.get());
}

float ModelFile::f32(const std::string & key) const {
    return gguf_get_val_f32(gguf_.get(), key_id(key, GGUF_TYPE_FLOAT32));
}

double ModelFile::f64(const std::string & key) const {
    return gguf_get_val_f64(gguf_.get(), key_id(key, GGUF_TYPE_FLOAT64));
}

bool ModelFile::boolean(const std::string & key) const {
    return gguf_get_val_bool(gguf_.get(), key_id(key, GGUF_TYPE_BOOL));
}

std::string ModelFile::str(const std::string & key) const {
    return gguf_get_val_str(gguf_.get(), key_id(key, GGUF_TYPE_STRING));
}

std::string ModelFile::one_of(const std::string & key, std::initializer_list<const char *> values) const {
    const std::string value = str(key);
    std::vector<std::string> known;
    for (const char * v : values) {
        if (value == v) return value;
        known.push_back(std::string("\"") + v + "\"");
    }
    throw file_error("the key " + key + " of " + path_ + " is \"" + value + "\", which " + layout_name() + " does not know; it knows " +
                     joined(known, known.size()) + "; " + remedy_);
}

std::vector<uint8_t> ModelFile::u8_array(const std::string & key) const {
    const int64_t id = key_id(key, GGUF_TYPE_ARRAY, GGUF_TYPE_UINT8);
    const uint8_t * data = (const uint8_t *) gguf_get_arr_data(gguf_.get(), id);
    return std::vector<uint8_t>(data, data + gguf_get_arr_n(gguf_.get(), id));
}

std::vector<int32_t> ModelFile::i32_array(const std::string & key) const {
    const int64_t id = key_id(key, GGUF_TYPE_ARRAY, GGUF_TYPE_INT32);
    const int32_t * data = (const int32_t *) gguf_get_arr_data(gguf_.get(), id);
    return std::vector<int32_t>(data, data + gguf_get_arr_n(gguf_.get(), id));
}

std::vector<float> ModelFile::f32_array(const std::string & key) const {
    const int64_t id = key_id(key, GGUF_TYPE_ARRAY, GGUF_TYPE_FLOAT32);
    const float * data = (const float *) gguf_get_arr_data(gguf_.get(), id);
    return std::vector<float>(data, data + gguf_get_arr_n(gguf_.get(), id));
}

std::vector<double> ModelFile::f64_array(const std::string & key) const {
    const int64_t id = key_id(key, GGUF_TYPE_ARRAY, GGUF_TYPE_FLOAT64);
    const double * data = (const double *) gguf_get_arr_data(gguf_.get(), id);
    return std::vector<double>(data, data + gguf_get_arr_n(gguf_.get(), id));
}

std::vector<std::string> ModelFile::str_array(const std::string & key) const {
    const int64_t id = key_id(key, GGUF_TYPE_ARRAY, GGUF_TYPE_STRING);
    std::vector<std::string> out(gguf_get_arr_n(gguf_.get(), id));
    for (size_t i = 0; i < out.size(); i++) out[i] = gguf_get_arr_str(gguf_.get(), id, i);
    return out;
}

uint64_t ModelFile::file_bytes() const {
    std::error_code error;
    const uint64_t size = std::filesystem::file_size(std::filesystem::u8path(path_), error);
    if (error) throw Error(Fault::Io, "cannot read the size of " + path_ + ": " + error.message());
    return size;
}

uint64_t ModelFile::weight_bytes() const {
    return weight_bytes([](const std::string &) { return true; });
}

uint64_t ModelFile::weight_bytes(const std::function<bool(const std::string & name)> & which) const {
    uint64_t bytes = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf_.get()); i++) {
        if (which(gguf_get_tensor_name(gguf_.get(), i))) bytes += gguf_get_tensor_size(gguf_.get(), i);
    }
    return bytes;
}

std::map<ggml_type, uint64_t> ModelFile::type_bytes() const {
    std::map<ggml_type, uint64_t> bytes;
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf_.get()); i++) {
        bytes[gguf_get_tensor_type(gguf_.get(), i)] += gguf_get_tensor_size(gguf_.get(), i);
    }
    return bytes;
}

size_t ModelFile::meta_count() const {
    return (size_t) gguf_get_n_kv(gguf_.get());
}

std::string ModelFile::meta_key(size_t index) const {
    return gguf_get_key(gguf_.get(), (int64_t) index);
}

std::string ModelFile::meta_json(size_t index) const {
    const gguf_context * g = gguf_.get();
    const int64_t id = (int64_t) index;
    const gguf_type type = gguf_get_kv_type(g, id);
    if (type == GGUF_TYPE_STRING) return json_string(gguf_get_val_str(g, id));
    if (type != GGUF_TYPE_ARRAY) return scalar_json(type, gguf_get_val_data(g, id));
    const gguf_type element = gguf_get_arr_type(g, id);
    const size_t n = gguf_get_arr_n(g, id);
    std::string out = "[";
    if (element == GGUF_TYPE_STRING) {
        for (size_t i = 0; i < n; i++) out += (i ? "," : "") + json_string(gguf_get_arr_str(g, id, i));
    } else if (element == GGUF_TYPE_ARRAY) {
        throw file_error("the key " + meta_key(index) + " of " + path_ + " holds arrays of arrays, which no layout has; " + remedy_);
    } else {
        const auto * data = (const uint8_t *) gguf_get_arr_data(g, id);
        const size_t width = gguf_scalar_size(element);
        for (size_t i = 0; i < n; i++) out += (i ? "," : "") + scalar_json(element, data + i * width);
    }
    return out + "]";
}

ModelIdentity read_identity(const ModelFile & file) {
    const auto text = [&](const std::string & key) {
        std::string value = file.str(key);
        if (value.empty()) throw file_error("the key " + key + " of " + file.path() + " is empty; " + file.remedy());
        return value;
    };
    const auto text_if_any = [&](const std::string & key) { return file.has(key) ? std::optional<std::string>(text(key)) : std::nullopt; };
    ModelIdentity id;
    id.name = text("general.name");
    id.organization = text("general.organization");
    id.basename = text("general.basename");
    id.size_label = text("general.size_label");
    id.finetune = text_if_any("general.finetune");
    id.version = text_if_any("general.version");
    id.license = text("general.license");
    id.repository = text("general.source.repo_url");
    const std::string url = text("general.source.url"), tree = id.repository + "/tree/";
    if (url.compare(0, tree.size(), tree) == 0) id.revision = url.substr(tree.size());
    if (id.revision.empty() || id.revision.find('/') != std::string::npos) {
        throw file_error("the key general.source.url of " + file.path() + " is " + url + ", where speech.cpp takes " + tree +
                         "<revision>, a revision of general.source.repo_url; " + file.remedy());
    }

    const uint32_t value = file.u32("general.file_type");
    const WeightType * named = nullptr;
    std::string known;
    for (const WeightType & t : weight_types()) {
        if (t.file_type == value) named = &t;
        known += (known.empty() ? "" : ", ") + std::to_string(t.file_type) + " (" + tensor_type_text(t.type) + ")";
    }
    if (!named) {
        throw file_error("the key general.file_type of " + file.path() + " is " + std::to_string(value) +
                         ", which names none of the weight types this release reads, " + known + "; " + file.remedy_for_unknown());
    }
    const std::map<ggml_type, uint64_t> bytes = file.type_bytes();
    const auto most = std::max_element(bytes.begin(), bytes.end(), [](const auto & a, const auto & b) { return a.second < b.second; });
    if (most == bytes.end() || most->first != named->type) {
        throw file_error("the key general.file_type of " + file.path() + " names " + tensor_type_text(named->type) + ", where " +
                         (most == bytes.end() ? std::string("the file holds no tensor") : "most of its tensors' bytes are " + tensor_type_text(most->first)) +
                         "; " + file.remedy());
    }
    id.weight_type = tensor_type_text(named->type);
    if (std::any_of(bytes.begin(), bytes.end(), [](const auto & b) { return ggml_is_quantized(b.first); })) file.u32("general.quantization_version");
    return id;
}

void check_model_keys(const ModelFile & file, const char * task, const char * language_use) {
    read_identity(file);
    const std::string file_task = file.one_of("speech.task", {"synthesis", "recognition", "detection", "diarization", "enhancement"});
    if (file_task != task) {
        throw file_error(file.path() + " says its speech.task is " + file_task + ", where its family does " + task + "; " + file.remedy());
    }
    if (file_task != "enhancement") file.size("speech.sample_rate");
    if ((file_task == "detection" || file_task == "diarization" || file_task == "enhancement") != !language_use) {
        throw std::logic_error(std::string("a family of ") + task + (language_use ? " gives a use of languages" : " gives no use of languages"));
    }
    if (!language_use) return;
    const std::string file_use = file.one_of("speech.language_use", {"steers", "checked"});
    if (file_use != language_use) {
        throw file_error(file.path() + " says its speech.language_use is " + file_use + ", where its family's is " + language_use + "; " +
                         file.remedy());
    }
    const std::vector<std::string> languages = file.str_array("general.languages");
    // BCP 47 names a language by its shortest ISO 639 code: two letters where ISO 639-1 has one, and three, from ISO 639-2
    // or 639-3, for a language it has none for, such as Cantonese (yue) and Filipino (fil). The GGUF specification asks
    // for two-letter codes, which those languages lack.
    const auto code = [](const std::string & l) {
        return (l.size() == 2 || l.size() == 3) && std::all_of(l.begin(), l.end(), [](char c) { return c >= 'a' && c <= 'z'; });
    };
    if (languages.empty() || !std::all_of(languages.begin(), languages.end(), code) ||
        std::adjacent_find(languages.begin(), languages.end(), std::greater_equal<std::string>()) != languages.end()) {
        throw file_error("general.languages of " + file.path() +
                         " is empty, holds what is not an ISO 639 code of two or three lowercase letters, or is not sorted without repeats; " +
                         file.remedy());
    }
}

void add_block(std::vector<TensorSpec> & tensors, const std::string & prefix, std::initializer_list<TensorSpec> parts) {
    for (const TensorSpec & part : parts) tensors.push_back({prefix + part.name, part.shape, part.storage});
}

void add_numbered(std::vector<TensorSpec> & tensors, const std::string & prefix, int count, std::initializer_list<TensorSpec> parts) {
    for (int i = 0; i < count; i++) {
        for (const TensorSpec & part : parts) {
            tensors.push_back({prefix + std::to_string(i) + (part.name.empty() ? "" : "." + part.name), part.shape, part.storage});
        }
    }
}
