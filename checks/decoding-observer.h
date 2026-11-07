#pragma once

// A check-side observer of the graphs a production backend computes, for the checks of a decoding that must be
// watched without touching it. It wraps a ggml backend by copying the backend's own struct and replacing the one
// entry a computation goes through, ggml_backend_i.graph_compute: the original's graph_compute runs first, the
// backend is synchronized, and only then are the tensors read that production itself marked as outputs
// (GGML_TENSOR_FLAG_OUTPUT), between one graph and the next, before the graph's memory is planned again. No graph,
// no output flag, no allocation and no tensor lifetime is changed, so what runs and what is marked is exactly what
// production built; this file only watches it go by.
//
// The wrapper needs ggml's struct ggml_backend, which ggml-backend-impl.h of the submodule defines; the check's
// CMake target puts ggml/src on its include path. The guid, the device and the private context are the original's
// own, so every other entry of the interface, which works from backend->context, behaves the same on the wrapper,
// and so do the functions that read the context directly, such as ggml_backend_cpu_set_threadpool(). The wrapper's
// free is a no-op: the tool that made the backend frees the backend itself, and the Observer frees the wrapper.

#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

#include "ggml-backend-impl.h"
#include "ggml.h"

namespace observe {

/** What one computed graph was, and the joint logits row it left, if it left exactly one. */
struct SeenGraph {
    enum class Kind {
        /** A full prediction step: the graph looks the label's embedding up through GGML_OP_GET_ROWS. */
        Step,
        /** The joint alone: no embedding lookup, and one output the size of the joint's logits. */
        Joint,
        /** Anything else, such as the encoder. */
        Other,
    };
    Kind kind;
    /**
     * How many of the graph's output nodes hold exactly the joint's logits, one contiguous row of that many float32
     * values: a prediction step and a joint graph leave exactly one, and a graph of either kind that leaves a
     * different number is a broken observation that the check fails on this count.
     */
    size_t matched = 0;
    /** The values of that one row, empty where the graph left none or more than one. */
    std::vector<float> logits;
};

class Observer;

/** The observers by their wrapped backends, so a trampoline without state of its own finds its observer. */
inline std::map<ggml_backend_t, Observer *> & observers() {
    static std::map<ggml_backend_t, Observer *> all;
    return all;
}

class Observer {
public:
    /**
     * Wraps `backend`, whose graphs computed from now on are seen, recognizing the joint's logits by their count
     * `outputs` (the tokens and the blank, then one a duration). `backend` stays the caller's to free.
     */
    Observer(ggml_backend_t backend, int64_t outputs) : backend_(backend), outputs_(outputs) {
        wrapper_ = (ggml_backend_t) std::malloc(sizeof(struct ggml_backend));
        if (!wrapper_) throw std::runtime_error("cannot allocate an observing backend");
        std::memcpy(wrapper_, backend, sizeof(struct ggml_backend));
        wrapper_->iface.graph_compute = &Observer::compute;
        // The tool frees the backend itself; a free of the wrapper would free the backend's context underneath it.
        wrapper_->iface.free = &Observer::no_free;
        observers()[wrapper_] = this;
    }

    ~Observer() {
        observers().erase(wrapper_);
        std::free(wrapper_);
    }

    Observer(const Observer &) = delete;
    Observer & operator=(const Observer &) = delete;

    /** The wrapped backend, to give wherever the real one was given. */
    ggml_backend_t backend() const { return wrapper_; }

    /** The graphs computed since this Observer was made or clear() was called, in the order they were computed. */
    const std::vector<SeenGraph> & seen() const { return seen_; }
    void clear() { seen_.clear(); }

    size_t count(SeenGraph::Kind kind) const {
        size_t n = 0;
        for (const SeenGraph & g : seen_) n += g.kind == kind;
        return n;
    }

private:
    static void no_free(ggml_backend_t) {}

    /**
     * The graph the backend was asked to compute, run by the backend's own graph_compute and then read: its kind by
     * the nodes it holds, and its logits row from the outputs production marked. Only a float32 output node marked
     * GGML_TENSOR_FLAG_OUTPUT that is one contiguous row whose leading axis, element count and byte count are all
     * exactly the joint's logits counts as a match; a padded or strided view of another shape would have
     * ggml_nbytes() larger than the floats it holds, and copying it would read past the check's vector. The row is
     * read only where exactly one node matches: a graph of a decoding kind that left no row, or more than one, is
     * kept with its count for the check to fail, for a label-embedding node alone does not make a decoder step whose
     * decisions were seen.
     */
    static ggml_status compute(ggml_backend_t self, struct ggml_cgraph * cgraph) {
        Observer * o = observers().at(self);
        const ggml_status status = o->backend_->iface.graph_compute(o->backend_, cgraph);
        ggml_backend_synchronize(o->backend_);
        if (status != GGML_STATUS_SUCCESS) return status;
        SeenGraph got;
        const size_t bytes = (size_t) o->outputs_ * sizeof(float);
        bool embedding = false;
        for (int i = 0; i < ggml_graph_n_nodes(cgraph); i++) {
            struct ggml_tensor * node = ggml_graph_node(cgraph, i);
            if (node->op == GGML_OP_GET_ROWS) embedding = true;
            if (node->type == GGML_TYPE_F32 && (node->flags & GGML_TENSOR_FLAG_OUTPUT) && node->ne[0] == o->outputs_ &&
                ggml_nelements(node) == o->outputs_ && ggml_nbytes(node) == bytes) {
                got.matched++;
                got.logits.resize((size_t) o->outputs_);
                ggml_backend_tensor_get(node, got.logits.data(), 0, bytes);
            }
        }
        if (got.matched != 1) got.logits.clear();
        got.kind = embedding ? SeenGraph::Kind::Step : got.matched == 1 ? SeenGraph::Kind::Joint : SeenGraph::Kind::Other;
        o->seen_.push_back(std::move(got));
        return status;
    }

    ggml_backend_t backend_ = nullptr;
    ggml_backend_t wrapper_ = nullptr;
    int64_t outputs_ = 0;
    std::vector<SeenGraph> seen_;
};

}  // namespace observe
