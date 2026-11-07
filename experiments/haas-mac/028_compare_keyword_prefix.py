#!/usr/bin/env python3
"""Compares the artifacts of the original, prefix-off and prefix-on processes of one recognition.

fastconformer-keyword-prefix-normal-check writes one artifact per process; experiments/haas-mac/028-keyword-prefix.md
says how the three are made. Each artifact is little-endian: the magic "SPKWP028", a u32 format version, sections of
a u32 tag, a u64 length and that many payload bytes, ended by tag 0 and nothing after it. Tag 1 is the request's
settings as JSON; tags 2 and 3 the observed and unobserved words (a speech byte and three u32-length UTF-8 strings);
tags 4 and 5 the observed and unobserved progress (a u64 count and that many f64); tag 6 the observed graph counts
and the row the boosted pass begins at; tag 7 the graph kinds in order (1 a prediction graph, 2 a joint-only one);
tag 8 every consumed joint logits row, its float32 bits unchanged.

The comparison enforces what the experiment approves and nothing looser: the three requests must be the same (the
metadata but the selector), the original and the prefix-off process must have been run with the selector unset or 0
and the prefix-on process with it 1 — a wrong build or a wrong --prefix in a role fails on its own, whatever else
coincides —, the observed and the unobserved run within a process must agree, the texts, the progress
bits and the logits bytes must be the same across the three processes, every value finite, and no argmax of the
tokens and the blank nor of the durations may move — those raw argmax values are decided before the tree's boost.
The ordinary graph kinds of the original and the prefix-off build must match, and the candidate with the prefix on
must have skipped at least one real boosted predictor prefix on this speech case: fewer prediction graphs and as
many more joint-only graphs after the boosted pass begins, the totals holding, for each cache hit still computes one
joint graph and a total count alone would show nothing. Every failure is printed; the artifacts are only read.

usage: 028_compare_keyword_prefix.py <original.kpw> <off.kpw> <on.kpw>
"""

import argparse
import json
import math
import struct
import sys

MAGIC = b"SPKWP028"
FORMAT = 1
(TAG_END, TAG_META, TAG_TEXT, TAG_QUIET_TEXT, TAG_PROGRESS, TAG_QUIET_PROGRESS, TAG_GRAPHS, TAG_KINDS,
 TAG_LOGITS) = range(9)
NO_BOUNDARY = (1 << 64) - 1
SHARED_META = ("model", "weights", "blank", "outputs", "duration_slots", "hidden", "alpha", "guard", "keywords",
               "decode_on", "backend", "device", "threads", "wav", "sample_rate", "samples", "frames", "decoding")


class ArtifactError(Exception):
    """An artifact that is not whole: truncated, malformed or of another format."""


class Words:
    """One process's words: the speech bit and the three texts."""

    def __init__(self, payload, what):
        if len(payload) < 1:
            raise ArtifactError(f"{what}: empty words section")
        self.speech = payload[0] != 0
        at = 1
        texts = []
        for _ in range(3):
            if at + 4 > len(payload):
                raise ArtifactError(f"{what}: a text length truncated")
            (size,) = struct.unpack_from("<I", payload, at)
            at += 4
            if at + size > len(payload):
                raise ArtifactError(f"{what}: a text truncated")
            texts.append(payload[at:at + size].decode("utf-8"))
            at += size
        if at != len(payload):
            raise ArtifactError(f"{what}: {len(payload) - at} bytes after the texts")
        self.plain, self.boosted, self.formatted = texts

    def same(self, other):
        return (self.speech == other.speech and self.plain == other.plain and self.boosted == other.boosted
                and self.formatted == other.formatted)


class Artifact:
    """One process's artifact, parsed and checked for wholeness."""

    def __init__(self, path):
        self.name = path
        with open(path, "rb") as f:
            data = f.read()
        if data[:8] != MAGIC:
            raise ArtifactError(f"{path}: not a keyword-prefix artifact (wrong magic)")
        at = 8
        (version,), at = self._scalar(data, at, "<I", 4, "the format version")
        if version != FORMAT:
            raise ArtifactError(f"{path}: format version {version}, this comparator reads {FORMAT}")
        sections = {}
        while True:
            if at == len(data):
                raise ArtifactError(f"{path}: truncated: no end section")
            (tag,), at = self._scalar(data, at, "<I", 4, "a section tag")
            (size,), at = self._scalar(data, at, "<Q", 8, "a section length")
            if at + size > len(data):
                raise ArtifactError(f"{path}: section {tag} truncated: {size} bytes wanted, {len(data) - at} left")
            payload = data[at:at + size]
            at += size
            if tag == TAG_END:
                if at != len(data):
                    raise ArtifactError(f"{path}: {len(data) - at} bytes after the end section")
                break
            if tag in sections:
                raise ArtifactError(f"{path}: section {tag} twice")
            sections[tag] = payload
        for tag in (TAG_META, TAG_TEXT, TAG_QUIET_TEXT, TAG_PROGRESS, TAG_QUIET_PROGRESS, TAG_GRAPHS, TAG_KINDS,
                    TAG_LOGITS):
            if tag not in sections:
                raise ArtifactError(f"{path}: section {tag} missing")
        self.meta = json.loads(sections[TAG_META].decode("utf-8"))
        for key in SHARED_META:
            if key not in self.meta:
                raise ArtifactError(f"{path}: the metadata holds no {key}")
        if "selector" not in self.meta:
            raise ArtifactError(f"{path}: the metadata holds no selector")
        if not (0 <= self.meta["blank"] < self.meta["outputs"]):
            raise ArtifactError(f"{path}: blank {self.meta['blank']} outside the joint's {self.meta['outputs']} outputs")
        self.text = Words(sections[TAG_TEXT], f"{path}: the observed words")
        self.quiet_text = Words(sections[TAG_QUIET_TEXT], f"{path}: the unobserved words")
        self.progress, self.progress_raw = self._progress(sections[TAG_PROGRESS], path, "the observed progress")
        self.quiet_progress, self.quiet_progress_raw = self._progress(sections[TAG_QUIET_PROGRESS], path,
                                                                      "the unobserved progress")
        graphs = sections[TAG_GRAPHS]
        if len(graphs) != 32:
            raise ArtifactError(f"{path}: the graph counts hold {len(graphs)} bytes, not four u64")
        self.steps, self.joints, self.others, self.boundary = struct.unpack("<QQQQ", graphs)
        kinds = sections[TAG_KINDS]
        (count,) = struct.unpack_from("<Q", kinds, 0)
        if len(kinds) != 8 + count:
            raise ArtifactError(f"{path}: the graph kinds hold {len(kinds) - 8} bytes for {count} graphs")
        self.kinds = kinds[8:]
        if any(kind not in (1, 2) for kind in self.kinds):
            raise ArtifactError(f"{path}: a graph kind the comparator knows not")
        rows = sections[TAG_LOGITS]
        (row_count,) = struct.unpack_from("<Q", rows, 0)
        self.outputs = self.meta["outputs"]
        if len(rows) != 8 + row_count * self.outputs * 4:
            raise ArtifactError(f"{path}: the logits hold {len(rows) - 8} bytes for {row_count} rows of "
                                f"{self.outputs} floats")
        self.rows = [rows[8 + i * self.outputs * 4:8 + (i + 1) * self.outputs * 4] for i in range(row_count)]
        if len(self.kinds) != len(self.rows) or self.steps + self.joints != len(self.rows) or self.others != 0:
            raise ArtifactError(f"{path}: {self.steps} prediction and {self.joints} joint-only graphs and "
                                f"{len(self.kinds)} kinds for {len(self.rows)} logits rows")
        if self.boundary != NO_BOUNDARY and self.boundary > len(self.rows):
            raise ArtifactError(f"{path}: the boosted pass begins past the last row")

    @staticmethod
    def _scalar(data, at, form, size, what):
        if at + size > len(data):
            raise ArtifactError(f"truncated at {what}")
        return struct.unpack_from(form, data, at), at + size

    @staticmethod
    def _progress(payload, path, what):
        (count,) = struct.unpack_from("<Q", payload, 0)
        if len(payload) != 8 + count * 8:
            raise ArtifactError(f"{path}: {what} holds {len(payload) - 8} bytes for {count} calls")
        raw = payload[8:]
        return list(struct.unpack(f"<{count}d", raw)), raw


def first_argmax(values):
    """The index of the largest value, the first of equal ones, as the decoding takes it."""
    best = 0
    for i in range(1, len(values)):
        if values[i] > values[best]:
            best = i
    return best


def section_counts(kinds, boundary):
    """The prediction and joint-only graphs of the boosted pass, from where it begins."""
    after = kinds[boundary:]
    return after.count(1), after.count(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("original", help="the artifact of the original-base build")
    parser.add_argument("off", help="the artifact of the prefix-off candidate")
    parser.add_argument("on", help="the artifact of the prefix-on candidate")
    args = parser.parse_args()
    if struct.pack("<I", 1) != b"\x01\x00\x00\x00":
        print("error: this comparator reads little-endian artifacts; run it on a little-endian host", file=sys.stderr)
        return 2

    try:
        artifacts = [Artifact(path) for path in (args.original, args.off, args.on)]
    except (ArtifactError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    original, off, on = artifacts

    fails = []

    def fail(message):
        print(f"FAIL {message}")
        fails.append(message)

    for key in SHARED_META:
        if not (original.meta[key] == off.meta[key] == on.meta[key]):
            fail(f"the requests differ in {key}: {original.meta[key]!r}, {off.meta[key]!r}, {on.meta[key]!r}")
    print(f"selectors: original {original.meta['selector']!r}, off {off.meta['selector']!r}, "
          f"on {on.meta['selector']!r}")
    for artifact, role, allowed in ((original, "the original-base build", ("unset", "0")),
                                    (off, "the prefix-off candidate", ("unset", "0")),
                                    (on, "the prefix-on candidate", ("1",))):
        if artifact.meta["selector"] not in allowed:
            fail(f"{artifact.name}: {role} ran with the requested selector {artifact.meta['selector']!r}, not one"
                 f" of {allowed}")

    for artifact in artifacts:
        if not artifact.text.same(artifact.quiet_text):
            fail(f"{artifact.name}: the observed and the unobserved run wrote different words")
        if artifact.progress_raw != artifact.quiet_progress_raw:
            fail(f"{artifact.name}: the observed and the unobserved run made different progress calls")
        if not artifact.text.speech:
            fail(f"{artifact.name}: the audio has no speech by the model's test, so no boosted pass ran")

    if not (original.text.same(off.text) and off.text.same(on.text)):
        fail("the three processes wrote different words")
    if not (original.progress_raw == off.progress_raw == on.progress_raw):
        fail("the three processes made different progress calls")
    if not (original.kinds[:original.boundary] == off.kinds[:off.boundary] == on.kinds[:on.boundary]):
        fail("the plain pass computed different graph kinds across the processes")

    for before, after in ((original, off), (off, on)):
        if len(before.rows) != len(after.rows):
            fail(f"{before.name} and {after.name} consumed different numbers of logits rows "
                 f"({len(before.rows)} and {len(after.rows)})")
            continue
        outputs, blank = before.outputs, before.meta["blank"]
        slots = outputs - blank - 1
        rows_differing = floats_differing = nonfinite = label_argmax = duration_argmax = 0
        max_abs = 0.0
        for a_row, b_row in zip(before.rows, after.rows):
            a = struct.unpack(f"<{outputs}f", a_row)
            b = struct.unpack(f"<{outputs}f", b_row)
            if a_row != b_row:
                rows_differing += 1
            for x, y in zip(a, b):
                if not (math.isfinite(x) and math.isfinite(y)):
                    nonfinite += 1
                elif x != y:
                    floats_differing += 1
                    max_abs = max(max_abs, abs(x - y))
            if first_argmax(a[:blank + 1]) != first_argmax(b[:blank + 1]):
                label_argmax += 1
            if slots > 0 and first_argmax(a[blank + 1:]) != first_argmax(b[blank + 1:]):
                duration_argmax += 1
        print(f"{before.name} against {after.name}: {len(before.rows)} rows, {rows_differing} rows differ in bytes "
              f"({floats_differing} floats, max |diff| {max_abs:.2e}), {nonfinite} values not finite, the argmax of "
              f"the tokens and the blank differs in {label_argmax} rows, the durations' in {duration_argmax}")
        if rows_differing:
            fail(f"{before.name} and {after.name} consumed joint logits that are not the same bytes, and no rounding "
                 f"difference is approved here")
        if nonfinite:
            fail(f"{before.name} and {after.name} hold {nonfinite} logits values that are not finite")
        if label_argmax or duration_argmax:
            fail(f"{before.name} and {after.name} hold rows whose raw argmax moved, before any boost")

    if original.kinds != off.kinds:
        fail("the original build and the prefix-off candidate computed different graph kinds for one recognition")
    if not (original.boundary == off.boundary == on.boundary) or on.boundary == NO_BOUNDARY:
        fail("the three processes disagree on where the boosted pass begins, or it never began")
    else:
        boundary = on.boundary
        before = (on.kinds[:boundary].count(1), on.kinds[:boundary].count(2))
        if before != (off.kinds[:boundary].count(1), off.kinds[:boundary].count(2)):
            fail("the plain pass computed different graphs in the candidate with the prefix on")
        off_after = section_counts(off.kinds, boundary)
        on_after = section_counts(on.kinds, boundary)
        skipped = off_after[0] - on_after[0]
        print(f"graphs: the plain pass {before[0]} prediction and {before[1]} joint-only; the boosted pass "
              f"{off_after[0]} and {off_after[1]} in original and off, {on_after[0]} and {on_after[1]} on")
        if skipped < 1:
            fail("the candidate with the prefix on skipped no boosted predictor prefix on this speech case")
        elif on_after[1] - off_after[1] != skipped:
            fail(f"the candidate with the prefix on skipped {skipped} prediction graphs but changed its joint-only "
                 f"graphs by {on_after[1] - off_after[1]}")
        elif sum(on_after) != sum(off_after):
            fail("each borrowed position still computes one joint graph, so the totals must hold")

    print(f"{len(fails)} failures" if fails else "the three artifacts agree on everything the experiment approves")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
