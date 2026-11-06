"""Build the independent 022 compiler variants with a pinned local toolchain.

Usage: build.py pin --out toolchain.json
       build.py build --source CLEAN_WORKTREE --toolchain toolchain.json --out NEW_BUILD
                      --variant baseline|022a|022b|022c-train|022c [--lto none|ggml|host]
                      [--profile profile.json]
       build.py pair --baseline build.json --candidate build.json
Pinning and building acquire the shared performance lock. No command downloads or pushes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
LOCK = '/private/tmp/speech-haas-mac-perf.lock'
BASE = '114c514430c8aa1455c077819f8e72e674a48b75'
GGML = 'e1a9ac12edbb57402618fd6936598603f5ee6266'
FP_FLAGS = ('-ffast-math', '-Ofast', '-ffp-contract=fast', '-fassociative-math', '-funsafe-math-optimizations')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'size_bytes': path.stat().st_size}


def command(*args):
    return subprocess.check_output([str(arg) for arg in args], text=True, stderr=subprocess.STDOUT).strip()


def git(source, *args):
    return command('git', '-C', source, *args)


def environment():
    forbidden = {'CC', 'CXX', 'AR', 'RANLIB', 'CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS', 'ASMFLAGS', 'ASFLAGS',
                 'OBJCFLAGS', 'OBJCXXFLAGS', 'ARCHFLAGS', 'SDKROOT',
                 'MACOSX_DEPLOYMENT_TARGET', 'CPATH', 'CPLUS_INCLUDE_PATH', 'C_INCLUDE_PATH', 'LIBRARY_PATH',
                 'CMAKE_PREFIX_PATH', 'CCC_OVERRIDE_OPTIONS', 'LLVM_PROFILE_FILE', 'DEVELOPER_DIR'}
    found = sorted(key for key in os.environ if key in forbidden or key.startswith(('GGML_', 'SPEECH_', 'DYLD_', 'CMAKE_')))
    if found:
        raise ValueError('Unset unrecorded compiler/runtime environment keys: ' + ', '.join(found))
    return dict(os.environ)


def toolchain():
    tools = {}
    hashes = {}
    for key, name in [('cc', 'clang'), ('cxx', 'clang++'), ('ld', 'ld'), ('ar', 'ar'), ('ranlib', 'ranlib'),
                      ('libtool', 'libtool'), ('profdata', 'llvm-profdata'), ('vtool', 'vtool'), ('nm', 'nm')]:
        path = Path(command('/usr/bin/xcrun', '--find', name)).absolute()
        real = str(path.resolve())
        if real not in hashes:
            hashes[real] = sha(path)
        tools[key] = {'path': str(path), 'sha256': hashes[real]}
    tools['cmake'] = artifact('/opt/homebrew/bin/cmake')
    sdk = Path(command('/usr/bin/xcrun', '--show-sdk-path')).resolve()
    settings = {p.name: sha(p) for p in sorted(sdk.glob('SDKSettings.*')) if p.is_file()}
    if not settings:
        raise ValueError('The selected SDK has no SDKSettings files to pin.')
    return {'tools': tools, 'clang_version': command(tools['cc']['path'], '--version'),
            'profdata_version': command(tools['profdata']['path'], '--version'),
            'cmake_version': command(tools['cmake']['path'], '--version'),
            'xcode_version': command('/usr/bin/xcodebuild', '-version'),
            'developer_directory': command('/usr/bin/xcode-select', '-p'),
            'sdk_path': str(sdk), 'sdk_version': command('/usr/bin/xcrun', '--show-sdk-version'),
            'sdk_settings_sha256': settings, 'macos': platform.mac_ver()[0], 'machine': platform.machine(),
            'deployment_target': '', 'generator': 'Unix Makefiles', 'arch': 'arm64',
            'python_executable': sys.executable, 'python_version': sys.version}


def verify_toolchain(path):
    pinned = read(path)
    if toolchain() != pinned['toolchain']:
        raise ValueError('The compiler, SDK, tools or host changed after the toolchain was pinned.')
    return pinned['toolchain']


def source_identity(source):
    if git(source, 'status', '--porcelain') or git(source / 'ggml', 'status', '--porcelain'):
        raise ValueError('Build provenance requires clean parent and ggml worktrees.')
    if git(source / 'ggml', 'rev-parse', 'HEAD') != GGML or git(source, 'rev-parse', 'HEAD:ggml') != GGML:
        raise ValueError('022 requires the original e1a9 ggml pin, without another kernel experiment.')
    if git(source, 'diff', '--name-only', BASE, 'HEAD', '--', 'src', 'include', 'ggml', 'CMakeLists.txt', 'tools', 'measure', 'vendor', 'VERSION'):
        raise ValueError('022 requires the baseline runtime and build source without another experiment.')
    return {'source_commit': git(source, 'rev-parse', 'HEAD'), 'source_tree': git(source, 'rev-parse', 'HEAD^{tree}'),
            'source_path': str(source), 'source_verified': True, 'ggml_commit': GGML}


def lto_mode(variant, requested):
    fixed = {'baseline': 'none', '022a': 'ggml', '022b': 'host'}
    if variant in fixed:
        if requested is not None:
            raise ValueError('--lto is only accepted for 022c and 022c-train.')
        return fixed[variant]
    if requested not in ('none', 'ggml', 'host'):
        raise ValueError('PGO builds require an explicit --lto none|ggml|host.')
    return requested


def configure_arguments(source, out, pinned, mode, variant, profile_path):
    flags = []
    if variant == '022c-train':
        flags = ['-fprofile-instr-generate', '-fprofile-update=atomic']
    elif variant == '022c':
        flags = ['-fprofile-instr-use=' + str(profile_path), '-Werror=profile-instr-out-of-date']
    args = [pinned['tools']['cmake']['path'], '-S', str(source), '-B', str(out), '-G', pinned['generator'],
            '-DCMAKE_BUILD_TYPE=Release', '-DGGML_NATIVE=ON', '-DGGML_METAL=ON', '-DGGML_METAL_EMBED_LIBRARY=ON',
            '-DGGML_METAL_SHADER_DEBUG=OFF', '-DGGML_CCACHE=OFF', '-DSPEECH_INSTALLED_GGML=OFF',
            '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DCMAKE_OSX_ARCHITECTURES=' + pinned['arch'],
            '-DCMAKE_OSX_SYSROOT=' + pinned['sdk_path'], '-DCMAKE_OSX_DEPLOYMENT_TARGET=' + pinned['deployment_target'],
            '-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG', '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG',
            '-DCMAKE_C_FLAGS=' + shlex.join(flags), '-DCMAKE_CXX_FLAGS=' + shlex.join(flags),
            '-DCMAKE_C_COMPILER=' + pinned['tools']['cc']['path'], '-DCMAKE_CXX_COMPILER=' + pinned['tools']['cxx']['path'],
            '-DCMAKE_AR=' + pinned['tools']['ar']['path'], '-DCMAKE_RANLIB=' + pinned['tools']['ranlib']['path'],
            '-DGGML_LTO=' + ('ON' if mode == 'ggml' else 'OFF'),
            '-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=' + ('ON' if mode == 'host' else 'OFF'),
            '-DCMAKE_PROJECT_speech-cpp_INCLUDE=' + str(HERE / 'check-ipo.cmake')]
    return args


def comparable_words(words, source, out):
    omitted = {'-flto=thin', '-fno-profile-instr-use', '-Werror=profile-instr-out-of-date'}
    return [word.replace(str(out), '<BUILD>').replace(str(source), '<SOURCE>') for word in words
            if word not in omitted and not word.startswith('-fprofile-instr-use=')]


def comparable_commands(commands, source, out):
    configure = []
    for word in commands[0]:
        if word.startswith(('-DGGML_LTO=', '-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=')):
            continue
        key, separator, value = word.partition('=')
        if separator and key in ('-DCMAKE_C_FLAGS', '-DCMAKE_CXX_FLAGS'):
            word = key + '=' + shlex.join(comparable_words(shlex.split(value), source, out))
        configure.append(word)
    return {'configure': comparable_words(configure, source, out),
            'build': comparable_words(commands[1], source, out)}


def audit_commands(source, out, mode, variant):
    source, out = source.resolve(), out.resolve()
    rows = read(out / 'compile_commands.json')
    counts = {'ggml': 0, 'host': 0, 'feature_detector_no_lto': 0, 'diagnostic_sources_without_profile_use': 0}
    compile_conditions = []
    for row in rows:
        file = Path(row['file']).resolve()
        words = row.get('arguments') or shlex.split(row['command'])
        compile_conditions.append({'file': comparable_words([str(file)], source, out)[0],
                                   'directory': comparable_words([str(Path(row['directory']).resolve())], source, out)[0],
                                   'arguments': comparable_words(words, source, out)})
        if file.suffix.lower() not in ('.c', '.cpp', '.cc', '.cxx', '.m', '.mm'):
            continue
        if any(flag in words for flag in FP_FLAGS):
            raise ValueError('A compile command enables an unapproved floating-point optimization.')
        if ('-fprofile-instr-generate' in words) != (variant == '022c-train'):
            raise ValueError('The generated instrumentation flags do not match the build variant.')
        if any(word.startswith('-fprofile-instr-use=') for word in words) != (variant == '022c'):
            raise ValueError('The generated profile-use flags do not match the build variant.')
        diagnostic = file.is_relative_to(source / 'checks') or file.is_relative_to(source / 'measure')
        if ('-fno-profile-instr-use' in words) != (variant == '022c' and diagnostic):
            raise ValueError('Only diagnostic translation units may omit profile use.')
        if '-fno-profile-instr-use' in words:
            if words.index('-fno-profile-instr-use') < next(i for i, word in enumerate(words) if word.startswith('-fprofile-instr-use=')):
                raise ValueError('The diagnostic profile override must follow the profile-use flag.')
            counts['diagnostic_sources_without_profile_use'] += 1
        scope = 'ggml' if file.is_relative_to(source / 'ggml') or file.is_relative_to(out / 'ggml') else 'host'
        expect = mode == 'host' or mode == 'ggml' and scope == 'ggml'
        if file.name == 'cpu-feats.cpp' and '-fno-lto' in words:
            counts['feature_detector_no_lto'] += 1
        elif ('-flto=thin' in words) != expect or any(w == '-flto' or w.startswith('-flto=full') for w in words):
            raise ValueError('The generated compile command does not match the intended ThinLTO scope: ' + str(file))
        counts[scope] += 1
    if not counts['ggml'] or not counts['host']:
        raise ValueError('The compile audit did not cover both ggml and parent targets.')
    links = {}
    for path in sorted(out.rglob('link.txt')):
        parts = path.relative_to(out).parts
        if any(part == 'CMakeFiles' and not parts[i + 1].endswith('.dir') for i, part in enumerate(parts[:-1])):
            continue
        text = path.read_text()
        if any(flag in shlex.split(text) for flag in FP_FLAGS):
            raise ValueError('A link command enables an unapproved floating-point optimization.')
        links[str(path.relative_to(out))] = text
    write(out / 'link-commands.json', links)
    for name in ('speech-cli', 'speech', 'fastconformer-timing'):
        text = links['CMakeFiles/' + name + '.dir/link.txt']
        if ('-flto=thin' in shlex.split(text)) != (mode == 'host'):
            raise ValueError('The final host link does not match the intended IPO scope: ' + name)
    return counts, {'compile': sorted(compile_conditions, key=lambda item: json.dumps(item, sort_keys=True)),
                    'link': {name: comparable_words(shlex.split(text), source, out) for name, text in links.items()}}


def build(args):
    env = environment()
    source, out = Path(args.source).resolve(), Path(args.out).resolve()
    if not HERE.is_relative_to(source):
        raise ValueError('Run the driver committed in the source worktree being built.')
    identity = source_identity(source)
    pinned = verify_toolchain(args.toolchain)
    mode = lto_mode(args.variant, args.lto)
    profile_path, profile = None, None
    if args.variant == '022c':
        if not args.profile:
            raise ValueError('022c requires the merged profile manifest.')
        profile = read(args.profile)
        for key in ('source_commit', 'ggml_commit'):
            if profile['instrumented_build'][key] != identity[key]:
                raise ValueError('The profile was trained on different source: ' + key)
        if profile['lto'] != mode or profile['toolchain_sha256'] != sha(args.toolchain):
            raise ValueError('PGO training and profile use require the same LTO and pinned toolchain.')
        profile_path = Path(profile['profile']['path']).resolve()
        if sha(profile_path) != profile['profile']['sha256']:
            raise ValueError('The merged profile changed after it was recorded.')
    elif args.profile:
        raise ValueError('--profile is only accepted for 022c.')
    out.mkdir(parents=True, exist_ok=False)
    configure = configure_arguments(source, out, pinned, mode, args.variant, profile_path)
    compile_command = [pinned['tools']['cmake']['path'], '--build', str(out), '--target', 'speech-cli', 'speech',
                       'fastconformer-timing', '-j4']
    with (out / 'build.log').open('x') as log:
        for cmd in (configure, compile_command):
            log.write(shlex.join(cmd) + '\n'); log.flush()
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    if identity != source_identity(source):
        raise ValueError('The source changed during the build.')
    audit, actual_conditions = audit_commands(source, out, mode, args.variant)
    conditions = {**comparable_commands([configure, compile_command], source, out), **actual_conditions}
    write(out / 'comparison-conditions.json', conditions)
    shaders = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('ggml-metal-embed-*.metal'))}
    if not shaders:
        raise ValueError('The build has no embedded shader source artifacts to compare.')
    manifest = {**identity, 'created_utc': datetime.now(timezone.utc).isoformat(), 'variant': args.variant, 'lto': mode,
                'instrumented': args.variant == '022c-train', 'performance_evaluation_eligible': args.variant != '022c-train',
                'toolchain_sha256': sha(args.toolchain), 'toolchain': pinned, 'driver_sha256': sha(__file__),
                'build_path': str(out), 'comparison_conditions': artifact(out / 'comparison-conditions.json'),
                'commands': [configure, compile_command], 'build_flags': {'configure': [a for a in configure if a.startswith('-D')],
                                                                      'build': compile_command[3:]},
                'cmake_cache_sha256': sha(out / 'CMakeCache.txt'), 'build_log_sha256': sha(out / 'build.log'),
                'compile_commands_sha256': sha(out / 'compile_commands.json'), 'link_commands_sha256': sha(out / 'link-commands.json'),
                'compile_scope': audit, 'embedded_shader_sha256': shaders,
                'stage_threads_argument': '<runs> <threads>' in (source / 'measure/fastconformer-timing.cpp').read_text(),
                'speech': artifact(out / 'speech'), 'stage_tool': artifact(out / 'fastconformer-timing'),
                'shared_library': artifact(out / 'libspeech.dylib')}
    manifest['mach_o_build_versions'] = {key: command(pinned['tools']['vtool']['path'], '-show-build', manifest[key]['path']).replace(manifest[key]['path'], '<BINARY>')
                                       for key in ('speech', 'stage_tool', 'shared_library')}
    exports = command(pinned['tools']['nm']['path'], '-gjU', manifest['shared_library']['path']).splitlines()
    if not exports or any(not symbol.startswith('_speech_') for symbol in exports):
        raise ValueError('The shared library exports symbols outside the C API.')
    manifest['shared_exports'] = exports
    if profile:
        manifest['pgo_profile'] = artifact(args.profile)
        manifest['profile'] = profile['profile']
    name = 'instrumented-build.json' if manifest['instrumented'] else 'build.json'
    write(out / name, manifest)
    print(out / name)


def verify_pair(baseline, candidate):
    conditions = []
    for item in (baseline, candidate):
        if item['instrumented'] or not item['performance_evaluation_eligible'] or not item['source_verified']:
            raise ValueError('Instrumented or unverified builds cannot be performance candidates.')
        configure, compile_command = item['commands']
        if item['build_flags'] != {'configure': [arg for arg in configure if arg.startswith('-D')], 'build': compile_command[3:]}:
            raise ValueError('The build flags disagree with the recorded commands.')
        reference = item['comparison_conditions']
        if sha(reference['path']) != reference['sha256']:
            raise ValueError('The generated comparison conditions changed after the build.')
        recorded = read(reference['path'])
        expected = comparable_commands(item['commands'], Path(item['source_path']), Path(item['build_path']))
        if any(recorded[key] != expected[key] for key in ('configure', 'build')):
            raise ValueError('The normalized conditions disagree with the build commands.')
        conditions.append(recorded)
    if conditions[0] != conditions[1]:
        raise ValueError('Configure, compile or link conditions differ beyond the intended LTO/PGO and path changes.')
    for key in ('source_commit', 'ggml_commit', 'toolchain_sha256', 'embedded_shader_sha256', 'shared_exports', 'mach_o_build_versions'):
        if baseline[key] != candidate[key]:
            raise ValueError('The comparison changes an uncontrolled condition: ' + key)
    if baseline['variant'] not in ('baseline', '022a', '022b'):
        raise ValueError('The baseline must not use PGO.')
    if candidate['variant'] == '022c':
        if baseline['lto'] != candidate['lto']:
            raise ValueError('PGO must compare the same source and LTO configuration.')
    elif baseline['variant'] != 'baseline' or candidate['variant'] not in ('022a', '022b'):
        raise ValueError('An LTO variant must compare against the non-LTO baseline.')


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--locked', action='store_true', help=argparse.SUPPRESS)
    sub = result.add_subparsers(dest='command', required=True)
    pin = sub.add_parser('pin'); pin.add_argument('--out', required=True)
    run = sub.add_parser('build')
    for name in ('source', 'toolchain', 'out'):
        run.add_argument('--' + name, required=True)
    run.add_argument('--variant', required=True, choices=('baseline', '022a', '022b', '022c-train', '022c'))
    run.add_argument('--lto', choices=('none', 'ggml', 'host')); run.add_argument('--profile')
    pair = sub.add_parser('pair')
    pair.add_argument('--baseline', required=True); pair.add_argument('--candidate', required=True)
    pair.add_argument('--assets'); pair.add_argument('--short')
    return result


def main():
    args = parser().parse_args()
    if args.command != 'pair' and not args.locked:
        subprocess.run(['/usr/bin/lockf', '-k', LOCK, sys.executable, '-B', __file__, '--locked', *sys.argv[1:]], check=True)
    elif args.command == 'pin':
        environment()
        write(args.out, {'captured_utc': datetime.now(timezone.utc).isoformat(), 'toolchain': toolchain()})
    elif args.command == 'build':
        build(args)
    else:
        baseline, candidate = read(args.baseline), read(args.candidate)
        verify_pair(baseline, candidate)
        if candidate['variant'] == '022c':
            if not args.assets or not args.short:
                raise ValueError('A PGO pair requires the actual evaluation --assets and --short manifests.')
            reference = candidate['pgo_profile']
            if sha(reference['path']) != reference['sha256']:
                raise ValueError('The PGO profile manifest changed after the build.')
            profile = read(reference['path'])
            if sha(args.assets) != profile['evaluation_assets']['sha256'] or sha(args.short) != profile['short_inputs']['sha256']:
                raise ValueError('PGO evaluation inputs differ from the sets excluded during training.')
        print('The build pair has matching source, toolchain, shaders and the intended LTO/PGO contrast.')


if __name__ == '__main__':
    main()
