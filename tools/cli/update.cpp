#include "commands.h"
#include "failure.h"
#include "speech.h"

namespace {

int run_update(const CommandLine &, FILE *) {
    throw Failure(speech_status_name(SPEECH_ERROR_UNSUPPORTED), "",
                  "uchkw/speech.cpp has no binary releases; update your fork checkout and rebuild from source: "
                  "https://github.com/uchkw/speech.cpp/blob/main/docs/install.md#update");
}

}  // namespace

Command update_command() {
    Command c;
    c.name = "update";
    c.usage = "update [--check]";
    c.summary = "explain how to update this source-only fork";
    c.description =
        "This fork has no binary releases. Both update and update --check fail with unsupported and point to the\n"
        "source build instructions, without checking for releases or running an installer. Messages go to stderr.";
    c.flags = {
        {"--check", "", false, "report that binary update checks are unsupported in this fork"},
    };
    c.run = run_update;
    return c;
}
