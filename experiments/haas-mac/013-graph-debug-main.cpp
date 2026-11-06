// Runs the 013 encoder check entry point with a log sink that forwards every level, Debug included, so that
// GGML_METAL_GRAPH_DEBUG=1 leaves the normal graph's node trace on stderr. The check's ordinary binary keeps the
// default sink, which drops Debug, so the variable alone shows nothing there. This launcher adds nothing else:
// no counter, no retained output, no command-buffer change, no environment rewrite. Its interface is the build
// alone: 013-encoder-check.cpp compiled with SPEECH_013_ENCODER_CHECK_NO_MAIN and linked with this file, whose
// own trace never enters the primary worker screen.
//
// usage: 013-graph-debug <model.gguf> [output.bin]

#include <cstdio>

#include "log.h"

int speech_013_encoder_check_main(int argc, char ** argv);

int main(int argc, char ** argv) {
    set_log_sink([](LogLevel level, const char * text) {
        (void) level;
        std::fprintf(stderr, "%s", text);
    });
    std::printf("graph-debug launcher: forwarding every log level to stderr\n");
    return speech_013_encoder_check_main(argc, argv);
}
