"""Pins the official source archives whose constants Haas's CPU converter reads without importing their kernels.

usage: python haas_sources.py <source folder>

Downloads sources separately from conversion, so conversion and its checkpoint verification need no network access.
The FLA archives and Haas commit are those of haas/uv.lock. None of these packages is installed or executed.
"""

import argparse
import ast
import hashlib
from pathlib import Path
import tarfile
import tomllib
import urllib.request


def source_pins():
    """Reads the full reference's source pins, with a digest for GitHub's archive of its Haas commit."""
    lock = tomllib.loads((Path(__file__).parent / "haas" / "uv.lock").read_text())
    packages = {package["name"]: package for package in lock["package"]}
    pins = {}
    for name in ("fla-core", "flash-linear-attention"):
        source = packages[name]["sdist"]
        pins[name] = {"url": source["url"], "sha256": source["hash"].removeprefix("sha256:"),
                      "file": source["url"].rsplit("/", 1)[1]}
    revision = packages["haas-asr"]["source"]["git"].rsplit("#", 1)[1]
    pins["haas-asr"] = {
        "url": f"https://codeload.github.com/third-intelligence/Haas-ASR/tar.gz/{revision}",
        "sha256": "ed92e81afae09697e8de2a4fef8317be2a4cd51c7560c8352d9ee4e02c7ce4a3",
        "file": f"Haas-ASR-{revision}.tar.gz",
    }
    return pins


class Sources:
    """Verified source text, used only to inspect constructor defaults and literal constants."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.pins = source_pins()
        self.files = {}
        for pin in self.pins.values():
            path = self.folder / pin["file"]
            if not path.is_file():
                raise FileNotFoundError(f"Download {pin['url']} to {path} before conversion")
            if hashlib.sha256(path.read_bytes()).hexdigest() != pin["sha256"]:
                raise ValueError(f"{path} differs from the pinned official source archive")

    def read(self, package, path):
        """Reads one source file from its verified package archive without extracting or importing it."""
        with tarfile.open(self.folder / self.pins[package]["file"]) as archive:
            members = {member.name.split("/", 1)[-1]: member for member in archive.getmembers()}
            data = archive.extractfile(members[path]).read()
        self.files[package + "/" + path] = hashlib.sha256(data).hexdigest()
        return data.decode("utf-8")


def definition(source, name):
    """Finds one named class or function in pinned Python source without executing the source."""
    node = ast.parse(source)
    for part in name.split("."):
        matches = [child for child in node.body if isinstance(child, (ast.ClassDef, ast.FunctionDef)) and child.name == part]
        if len(matches) != 1:
            raise ValueError(f"The official source does not define exactly one {name}")
        node = matches[0]
    return node


def defaults(source, name):
    """Reads a pinned function's literal defaults; nonliteral defaults are deliberately not evaluated."""
    args = definition(source, name).args
    pairs = list(zip(args.args[-len(args.defaults):], args.defaults)) if args.defaults else []
    pairs += list(zip(args.kwonlyargs, args.kw_defaults))
    return {arg.arg: ast.literal_eval(value) for arg, value in pairs if value is not None}


def assignment(source, name):
    """Finds a unique top-level constant or attribute assignment in a pinned source fragment."""
    matches = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(ast.unparse(target) == name for target in node.targets):
            matches.append(node.value)
    if len(matches) != 1:
        raise ValueError(f"The official source does not assign exactly one {name}")
    return matches[0]


def literal(source, name):
    """Reads a literal constant from pinned source."""
    return ast.literal_eval(assignment(source, name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    for pin in source_pins().values():
        path = args.folder / pin["file"]
        if path.exists():
            data = path.read_bytes()
        else:
            with urllib.request.urlopen(pin["url"], timeout=60) as response:
                data = response.read()
        if hashlib.sha256(data).hexdigest() != pin["sha256"]:
            raise ValueError(f"The download for {path} differs from the pinned source")
        path.write_bytes(data)
        print(path)


if __name__ == "__main__":
    main()
