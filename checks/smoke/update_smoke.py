"""Checks that source builds cannot invoke a binary updater or the shell installer on POSIX, without network access.

No models or GPU are needed. Windows binaries, PowerShell and terminal update notices are not checked.

usage: python3 checks/smoke/update_smoke.py <speech> <work dir>
"""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def check_source_updates(speech, work):
    """Runs the source-only entry points with an isolated home and a recording curl substitute on POSIX."""
    if os.name != "posix":
        raise SystemExit("update_smoke.py needs POSIX sh; Windows is not checked")
    executable = shutil.which(speech)
    if executable is None:
        raise SystemExit(f"cannot find executable {speech!r}; give its path or add its directory to PATH")
    speech = Path(executable).resolve()
    Path(work).mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="source-updates-", dir=Path(work).resolve()))
    home, fake = folder / "home", folder / "bin"
    installed = home / ".local/share/speech.cpp/1.0.0/speech"
    installed.parent.mkdir(parents=True)
    fake.mkdir()
    shutil.copy2(speech, installed)
    before = hashlib.sha256(installed.read_bytes()).digest()
    curl = fake / "curl"
    curl.write_text('''#!/bin/sh
printf '%s\\n' "$*" >> "$SPEECH_TEST_CURL_LOG"
case "$*" in
    */releases/latest*) printf '%s\\n' 'https://github.com/nyosegawa/speech.cpp/releases/tag/v999.0.0' ;;
    */install.sh*) printf '%s\\n' '#!/bin/sh' 'printf "%s\\n" invoked >> "$SPEECH_TEST_INSTALLER_LOG"' ;;
    *) exit 97 ;;
esac
''')
    curl.chmod(0o755)
    env = {**os.environ, "HOME": str(home), "XDG_CACHE_HOME": str(home / ".cache"),
           "PATH": str(fake) + os.pathsep + os.environ["PATH"], "NO_COLOR": "1",
           "SPEECH_TEST_CURL_LOG": str(folder / "curl.log"),
           "SPEECH_TEST_INSTALLER_LOG": str(folder / "installer.log")}
    for executable in (installed, speech):
        for args in (("update",), ("update", "--check")):
            r = subprocess.run([str(executable), *args], env=env, capture_output=True, timeout=20)
            assert r.returncode == 1 and r.stderr.startswith(b"speech: unsupported:"), (str(executable), args, r.returncode, r.stderr)
            assert not r.stdout, r.stdout
    installer = Path(__file__).resolve().parents[2] / "install.sh"
    for args in ((), ("--version", "0.7.0", "--no-modify-path")):
        r = subprocess.run(["/bin/sh", str(installer), *args], env=env, capture_output=True, timeout=20)
        assert r.returncode != 0 and r.stderr and not r.stdout, (args, r.returncode, r.stdout, r.stderr)
    assert not (folder / "curl.log").exists(), "a binary update entry point invoked curl"
    assert not (folder / "installer.log").exists(), "speech invoked the downloaded installer"
    assert hashlib.sha256(installed.read_bytes()).digest() == before, "speech replaced its executable"
    print(f"source-only update and installer checks pass; fixtures: {folder}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__.strip().splitlines()[-1])
    check_source_updates(*sys.argv[1:])
