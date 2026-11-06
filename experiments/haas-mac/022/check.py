"""Check 022 recipe and split guards without configuring, compiling, hashing audio or running a model.

Usage: /usr/bin/python3 -B experiments/haas-mac/022/check.py
"""
import copy
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
import build
import pgo


def rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('An invalid comparison or split was accepted.')


def recipes():
    pinned = {'tools': {name: {'path': '/tools/' + name} for name in ('cmake', 'cc', 'cxx', 'ar', 'ranlib')},
              'generator': 'Unix Makefiles', 'arch': 'arm64', 'sdk_path': '/sdk', 'deployment_target': ''}
    for variant, requested, expected in [('baseline', None, 'none'), ('022a', None, 'ggml'), ('022b', None, 'host'),
                                          ('022c-train', 'host', 'host'), ('022c', 'none', 'none')]:
        mode = build.lto_mode(variant, requested)
        assert mode == expected
        args = build.configure_arguments(Path('/source'), Path('/build'), pinned, mode, variant, Path('/profile/data'))
        assert '-DGGML_NATIVE=ON' in args
        assert '-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG' in args
        assert '-DGGML_METAL_EMBED_LIBRARY=ON' in args and '-DGGML_METAL_SHADER_DEBUG=OFF' in args
        assert '-DGGML_LTO=' + ('ON' if mode == 'ggml' else 'OFF') in args
        assert '-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=' + ('ON' if mode == 'host' else 'OFF') in args
        text = '\n'.join(args)
        assert ('-fprofile-instr-generate' in text) == (variant == '022c-train')
        assert ('-fprofile-instr-use=' in text) == (variant == '022c')
        assert not any(flag in text for flag in build.FP_FLAGS)
    rejects(lambda: build.lto_mode('baseline', 'host'))
    rejects(lambda: build.lto_mode('022c', None))


def pairs():
    base = {'source_commit': 'source', 'ggml_commit': 'ggml', 'toolchain_sha256': 'toolchain',
            'embedded_shader_sha256': {'shader': 'same'}, 'shared_exports': ['_speech_model_load'],
            'mach_o_build_versions': {'speech': 'same'}, 'source_verified': True,
            'instrumented': False, 'performance_evaluation_eligible': True, 'variant': 'baseline', 'lto': 'none'}
    build.verify_pair(base, {**base, 'variant': '022a', 'lto': 'ggml'})
    build.verify_pair(base, {**base, 'variant': '022b', 'lto': 'host'})
    build.verify_pair({**base, 'variant': '022b', 'lto': 'host'}, {**base, 'variant': '022c', 'lto': 'host'})
    rejects(lambda: build.verify_pair(base, {**base, 'variant': '022c', 'lto': 'host'}))
    rejects(lambda: build.verify_pair(base, {**base, 'variant': '022a', 'instrumented': True}))
    rejects(lambda: build.verify_pair(base, {**base, 'variant': '022a', 'performance_evaluation_eligible': False}))
    for key in ('source_commit', 'ggml_commit', 'toolchain_sha256', 'embedded_shader_sha256', 'shared_exports', 'mach_o_build_versions'):
        rejects(lambda key=key: build.verify_pair(base, {**base, 'variant': '022a', key: 'different'}))


def compiler_audits():
    def run(variant, mode, defect=None):
        with tempfile.TemporaryDirectory(prefix='haas-022-metadata-') as directory:
            root = Path(directory); source = root / 'source'; out = root / 'build'; out.mkdir()
            rows = []
            for name in ('src/model.cpp', 'ggml/src/model.cpp', 'measure/tool.cpp'):
                words = ['clang++', '-O3']
                if mode == 'host' or mode == 'ggml' and name.startswith('ggml/'):
                    words.append('-flto=thin')
                if variant == '022c-train':
                    words += ['-fprofile-instr-generate', '-fprofile-update=atomic']
                if variant == '022c':
                    words.append('-fprofile-instr-use=/profile')
                    if name.startswith('measure/'):
                        words.append('-fno-profile-instr-use')
                if defect == 'fp' and name.startswith('src/'):
                    words.append('-ffast-math')
                if defect == 'profile-leak' and name.startswith('src/'):
                    words.append('-fprofile-instr-generate')
                if defect == 'diagnostic' and name.startswith('measure/'):
                    words.remove('-fno-profile-instr-use')
                if defect == 'production' and name.startswith('src/'):
                    words.append('-fno-profile-instr-use')
                rows.append({'file': str(source / name), 'arguments': words})
            build.write(out / 'compile_commands.json', rows)
            for target in ('speech-cli', 'speech', 'fastconformer-timing'):
                path = out / 'CMakeFiles' / (target + '.dir'); path.mkdir(parents=True)
                (path / 'link.txt').write_text('clang++' + (' -flto=thin' if mode == 'host' and defect != 'link' else ''))
            return build.audit_commands(source, out, mode, variant)
    for variant, mode in [('baseline', 'none'), ('022a', 'ggml'), ('022b', 'host'),
                          ('022c-train', 'host'), ('022c', 'host')]:
        assert run(variant, mode)['host'] == 2
    rejects(lambda: run('baseline', 'none', 'fp'))
    rejects(lambda: run('baseline', 'none', 'profile-leak'))
    rejects(lambda: run('022c', 'host', 'diagnostic'))
    rejects(lambda: run('022c', 'host', 'production'))
    rejects(lambda: run('022b', 'host', 'link'))


def splits():
    records = [{'file': f'train-{i}.wav', 'source_recording_id': f'train-{i}.wav', 'id': f'train-{i}',
                'pcm_sha256': f'{i:064x}', 'samples': seconds * 16000, 'seconds': seconds,
                'source_samples': seconds * 16000, 'start_sample': 0, 'stratum': stratum, 'reference': 'fixture'}
               for i, (seconds, stratum) in enumerate([(5, '3-10s')] * 3 + [(15, '10-20s')] * 3 + [(25, '20-30s')] * 3)]
    training = {'dataset': {'repository': 'google/fleurs', 'revision': pgo.REVISION, 'config': 'ja_jp', 'split': 'train',
                            'archive_sha256': 'a' * 64, 'metadata_sha256': 'b' * 64,
                            'archive_url': 'https://example.test/train.tar.gz', 'metadata_url': 'https://example.test/train.tsv'},
                'sample_rate': 16000, 'format': 'mono PCM16 little endian', 'records': records}
    assets = {'corpus': [{'file': 'eval.wav', 'id': 'eval-id', 'pcm_sha256': 'c' * 64}], 'subset': [], 'stages': []}
    short = {'records': [{'file': 'prefix.pcm', 'source_file': 'prefix-parent.wav',
                          'pcm_sha256': 'd' * 64, 'source_pcm_sha256': 'e' * 64}]}
    assert pgo.validate_split(training, assets, short)['recordings'] == 9
    for key, value in [('file', 'eval.wav'), ('source_recording_id', 'prefix-parent.wav'), ('id', 'eval-id'),
                       ('pcm_sha256', 'c' * 64), ('pcm_sha256', 'd' * 64), ('pcm_sha256', 'e' * 64),
                       ('file', 'train-1.wav'), ('source_recording_id', 'train-1.wav'), ('pcm_sha256', f'{1:064x}'),
                       ('start_sample', 1), ('source_samples', 160000), ('seconds', 4), ('stratum', '10-20s')]:
        invalid = copy.deepcopy(training); invalid['records'][0][key] = value
        rejects(lambda: pgo.validate_split(invalid, assets, short))
    for key, value in [('split', 'test'), ('revision', 'unfixed'), ('archive_sha256', ''), ('config', 'en_us')]:
        invalid = copy.deepcopy(training); invalid['dataset'][key] = value
        rejects(lambda: pgo.validate_split(invalid, assets, short))
    invalid = copy.deepcopy(training); invalid['records'].pop()
    rejects(lambda: pgo.validate_split(invalid, assets, short))


if __name__ == '__main__':
    recipes(); pairs(); compiler_audits(); splits()
    print('022 recipe, comparison and disjoint-training metadata checks pass; no build or inference runs.')
