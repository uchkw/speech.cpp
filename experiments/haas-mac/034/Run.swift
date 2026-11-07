// Runs the normal Core ML artifact serially or records its anticipated compute plan.
// Usage: Run MANIFEST f32|f16 CPU_ONLY|CPU_AND_GPU|CPU_AND_NE|ALL NEW_OUTPUT_DIR plan|predict|trace ROUNDS
// Compile the one file with swiftc -O -parse-as-library. Prediction and plan construction require
// an assigned campaign compute lease.

import CoreML
import CryptoKit
import Darwin
import Foundation
import os

struct Failure: Error, CustomStringConvertible {
    let description: String
}

func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw Failure(description: message) }
}

func digest(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func fileDigest(_ url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var hasher = SHA256()
    while let chunk = try handle.read(upToCount: 4 * 1024 * 1024), !chunk.isEmpty { hasher.update(data: chunk) }
    return hasher.map { String(format: "%02x", $0) }.joined()
}

func packageIdentity(_ package: URL) throws -> [String: String] {
    // Streams the package the same way the Python side's inventory does, every file's relative
    // path to its SHA-256, so a swapped or edited package cannot pass as the checked one.
    let base = package.standardizedFileURL.resolvingSymlinksInPath()
    var files = [URL]()
    var pending = [base]
    while let current = pending.popLast() {
        for child in try FileManager.default.contentsOfDirectory(at: current, includingPropertiesForKeys: [.isDirectoryKey, .isRegularFileKey]) {
            let values = try child.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey])
            if values.isDirectory == true { pending.append(child) }
            else if values.isRegularFile == true { files.append(child) }
        }
    }
    try require(!files.isEmpty, "The model package holds no files: " + base.path)
    var identity = [String: String]()
    for file in files { identity[String(file.path.dropFirst(base.path.count + 1))] = try fileDigest(file) }
    return identity
}

func hardwareName() throws -> String {
    var count = 0
    try require(sysctlbyname("machdep.cpu.brand_string", nil, &count, nil, 0) == 0 && count > 0, "Cannot read hardware identity.")
    var bytes = [CChar](repeating: 0, count: count)
    try require(sysctlbyname("machdep.cpu.brand_string", &bytes, &count, nil, 0) == 0, "Cannot read hardware identity.")
    return String(cString: bytes)
}

func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
func seconds(_ start: UInt64) -> Double { Double(now() - start) / 1e9 }

func writeJSON(_ value: Any, _ url: URL) throws {
    try require(!FileManager.default.fileExists(atPath: url.path), "Output already exists: " + url.path)
    try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys]).write(to: url)
}

struct Shape: Decodable, Equatable {
    let id: String
    let t, u, p, q: Int
}

struct Input: Decodable {
    let id: String
    let t, u, p, q: Int
    let features_raw, features_sha256: String
    let seconds: Double
    var shape: Shape { Shape(id: id, t: t, u: u, p: p, q: q) }
}

struct Pins: Decodable { let cases: [Shape] }
struct Manifest: Decodable {
    let schema: Int
    let experiment, shape_set: String
    let pins: Pins
    let cases: [Input]
    let models: [String: [String: String]]
}

func inputProvider(_ item: Input) throws -> MLDictionaryFeatureProvider {
    let bytes = try Data(contentsOf: URL(fileURLWithPath: item.features_raw))
    try require(digest(bytes) == item.features_sha256 && bytes.count == 128 * item.p * 4, "Input bytes changed: " + item.id)
    var values = [Float](repeating: 0, count: 128 * item.p)
    values.withUnsafeMutableBytes { destination in bytes.copyBytes(to: destination) }
    try require(values.allSatisfy { $0.isFinite }, "Nonfinite features: " + item.id)
    let features = MLMultiArray(MLShapedArray<Float>(scalars: values, shape: [1, 128, item.p]))
    let length = MLMultiArray(MLShapedArray<Int32>(scalars: [Int32(item.t)], shape: [1]))
    return try MLDictionaryFeatureProvider(dictionary: ["features": features, "length": length])
}

func projection(_ output: MLFeatureProvider, _ item: Input) throws -> [Float] {
    guard let joint = output.featureValue(for: "joint_encoded")?.multiArrayValue,
          let length = output.featureValue(for: "encoded_length")?.multiArrayValue else {
        throw Failure(description: "The normal artifact must return projection and length.")
    }
    try require(joint.dataType == .float32 && joint.shape.map { $0.intValue } == [1, item.q, 640], "Projection shape or type differs.")
    try require(length.dataType == .int32 && length.shape.map { $0.intValue } == [1], "Length type or shape differs.")
    try require(MLShapedArray<Int32>(length).scalars == [Int32(item.u)], "The model returned the wrong valid length.")
    let full = MLShapedArray<Float>(joint).scalars
    try require(full.allSatisfy { $0.isFinite }, "Nonfinite padded projection output.")
    return Array(full.prefix(item.u * 640))
}

@available(macOS 14.4, *)
func computePlan(_ compiled: URL, _ configuration: MLModelConfiguration) async throws -> [[String: Any]] {
    let plan = try await MLComputePlan.load(contentsOf: compiled, configuration: configuration)
    guard case let .program(program) = plan.modelStructure else {
        throw Failure(description: "The compiled artifact is not an ML Program.")
    }
    var rows = [[String: Any]]()
    func block(_ value: MLModelStructure.Program.Block, _ path: String) {
        for (index, operation) in value.operations.enumerated() {
            let location = path + "/" + String(index)
            let usage = plan.deviceUsage(for: operation)
            let cost = plan.estimatedCost(of: operation)
            rows.append(["path": location, "operator": operation.operatorName,
                         "outputs": operation.outputs.map { $0.name },
                         "preferred": usage.map { String(describing: $0.preferred) as Any } ?? NSNull(),
                         "supported": usage.map { $0.supported.map { String(describing: $0) } as Any } ?? NSNull(),
                         "estimated_relative_cost": cost.map { $0.weight as Any } ?? NSNull()])
            for (nested, child) in operation.blocks.enumerated() { block(child, location + "/block" + String(nested)) }
        }
    }
    for name in program.functions.keys.sorted() { block(program.functions[name]!.block, name) }
    return rows
}

@main
struct Run {
    /// The result directory this invocation created, with its mode; only a directory this
    /// invocation created may ever hold its failure record.
    private static var owned: (url: URL, mode: String)?

    @available(macOS 14.4, *)
    static func main() async throws {
        do {
            try await execute(CommandLine.arguments)
        } catch {
            // An existing directory — a successful run's or an unrelated one — is left untouched;
            // only a fresh run that fails keeps its evidence, and writing the record must not mask
            // the error that caused it.
            if let output = owned {
                try? writeJSON(["stage": "native", "mode": output.mode, "error": String(describing: error)],
                               output.url.appendingPathComponent("failure.json"))
            }
            throw error
        }
    }

    @available(macOS 14.4, *)
    static func execute(_ args: [String]) async throws {
        try require(args.count == 7, "Usage: Run MANIFEST PRECISION UNITS OUT plan|predict|trace ROUNDS")
        let precision = args[2], unitsName = args[3], mode = args[5]
        let units: [String: MLComputeUnits] = ["CPU_ONLY": .cpuOnly, "CPU_AND_GPU": .cpuAndGPU,
                                               "CPU_AND_NE": .cpuAndNeuralEngine, "ALL": .all]
        guard let computeUnits = units[unitsName], let rounds = Int(args[6]) else {
            throw Failure(description: "Unknown compute units or invalid round count.")
        }
        try require(["f32", "f16"].contains(precision) && ["plan", "predict", "trace"].contains(mode), "Unknown precision or mode.")
        try require(rounds >= (mode == "predict" ? 3 : 1), "Prediction needs at least three rounds.")
        let source = try Data(contentsOf: URL(fileURLWithPath: args[1]))
        let manifest = try JSONDecoder().decode(Manifest.self, from: source)
        try require(manifest.schema == 1 && manifest.experiment == "034" && ["fixed", "enumerated"].contains(manifest.shape_set), "Unknown manifest.")
        let wanted = manifest.pins.cases.filter { manifest.shape_set == "enumerated" || $0.p == 1008 }
        try require(manifest.cases.map { $0.shape } == wanted && !wanted.isEmpty, "The case list differs from its pins.")
        for item in manifest.cases {
            try require(item.t > 0 && item.t <= item.p && item.u > 0 && item.u <= item.q && item.seconds > 0, "Invalid input extent.")
            try require(FileManager.default.fileExists(atPath: item.features_raw), "Missing input bytes: " + item.id)
        }
        guard let modelPath = manifest.models[precision]?["normal"] else { throw Failure(description: "Missing normal model.") }
        let package = URL(fileURLWithPath: modelPath).standardizedFileURL.resolvingSymlinksInPath()
        try require(FileManager.default.fileExists(atPath: package.path), "The manifest's normal package is missing: " + modelPath)
        let output = URL(fileURLWithPath: args[4]).standardizedFileURL.resolvingSymlinksInPath()
        try require(output.pathComponents.contains("outputs") && !FileManager.default.fileExists(atPath: output.path), "Use a new directory under outputs/.")
        let sourceDirectory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().standardizedFileURL.resolvingSymlinksInPath()
        try require(!output.path.hasPrefix(sourceDirectory.path + "/"), "Artifacts cannot be written beside experiment source.")
        // Everything above only reads; the disk is touched from here on. The package identity is
        // bound at compilation time, outside every timing scope below.
        let identity = try packageIdentity(package)
        try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        do {
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
        } catch {
            throw Failure(description: "The result directory already exists: " + output.path)
        }
        owned = (output, mode)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        try writeJSON(["schema": 1, "manifest_sha256": digest(source), "model": modelPath,
                       "model_identity": identity, "precision": precision,
                       "compute_units": unitsName, "mode": mode, "rounds": rounds,
                       "hardware": try hardwareName(), "runner_sha256": digest(try Data(contentsOf: URL(fileURLWithPath: args[0]))),
                       "os": ProcessInfo.processInfo.operatingSystemVersionString,
                       "scope": "Normal encoder+joint.enc artifact only; no worker or full request timing."], output.appendingPathComponent("run.json"))
        let compileStart = now()
        let temporary = try MLModel.compileModel(at: package)
        let compileSeconds = seconds(compileStart)
        // The temporary compiled copy is removed as soon as the retained copy exists, so the
        // two-copy peak lasts only through the copy.
        defer { try? FileManager.default.removeItem(at: temporary) }
        let compiled = output.appendingPathComponent("normal.mlmodelc")
        try FileManager.default.copyItem(at: temporary, to: compiled)
        if mode == "plan" {
            let rows = try await computePlan(compiled, configuration)
            try writeJSON(["compile_seconds": compileSeconds, "operations": rows,
                           "scope": "Anticipated device usage and relative cost; not actual per-shape execution."], output.appendingPathComponent("plan.json"))
            return
        }
        let loadStart = now()
        let model = try MLModel(contentsOf: compiled, configuration: configuration)
        let loadSeconds = seconds(loadStart)
        try require(Set(model.modelDescription.outputDescriptionsByName.keys) == Set(["joint_encoded", "encoded_length"]),
                    "Use the normal artifact with only projection and length outputs.")
        let inputStart = now()
        let providers = try manifest.cases.map { try inputProvider($0) }
        let inputSeconds = seconds(inputStart)
        let log = OSLog(subsystem: "speech.cpp.haas.034", category: "prediction")
        var records = [[String: Any]]()
        var firstOutputs = [String: [Float]]()
        for round in 0...rounds {
            let shift = round == 0 ? 0 : (round * 7) % manifest.cases.count
            for step in 0..<manifest.cases.count {
                let index = (step + shift) % manifest.cases.count
                let item = manifest.cases[index]
                let identifier = OSSignpostID(log: log)
                if mode == "trace" { os_signpost(.begin, log: log, name: "Prediction", signpostID: identifier, "%{public}@", item.id as NSString) }
                let start = now()
                let predicted = try model.prediction(from: providers[index])
                let predictionSeconds = seconds(start)
                if mode == "trace" { os_signpost(.end, log: log, name: "Prediction", signpostID: identifier) }
                let readStart = now()
                let values = try projection(predicted, item)
                let readSeconds = seconds(readStart)
                if let first = firstOutputs[item.id] {
                    try require(values.map { $0.bitPattern } == first.map { $0.bitPattern }, "Repeated projection changed: " + item.id)
                } else {
                    firstOutputs[item.id] = values
                    try values.withUnsafeBufferPointer { try Data(buffer: $0).write(to: output.appendingPathComponent(item.id + ".joint.f32")) }
                }
                records.append(["id": item.id, "p": item.p, "t": item.t, "u": item.u, "round": round,
                                "phase": round == 0 ? "first_visit" : "repeat", "prediction_seconds": predictionSeconds,
                                "host_output_seconds": readSeconds, "audio_seconds": item.seconds])
            }
        }
        try writeJSON(["compile_seconds": compileSeconds, "load_seconds": loadSeconds, "input_prepare_seconds": inputSeconds,
                       "records": records, "trace_enabled": mode == "trace",
                       "timing_scope": "Synchronous normal-model prediction API only; host preparation/readback reported separately.",
                       "actual_ane_execution": "Requires correlated Core ML and Neural Engine/GPU Instruments capture."], output.appendingPathComponent("result.json"))
    }
}
