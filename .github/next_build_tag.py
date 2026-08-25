"""Pick the next wheel build tag for a TestPyPI upload.

TestPyPI never allows a filename to be reused, not even after the file is
deleted, so re-uploading an unchanged version always needs a fresh filename.
A wheel build tag (PEP 427) provides one without touching the version: the
same 0.1.0 wheel can go up as 0.1.0-1, then 0.1.0-2, and pip resolves the
highest build tag as the newest.

This reads the version from the wheels that are about to be uploaded, asks the
index which build tags that version already uses, and prints the next one in
GITHUB_OUTPUT format.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
import uuid

# {name}-{version}[-{build}]-{python}-{abi}-{platform}.whl
WHEEL_RE = re.compile(
    r"^(?P<name>[^-]+)-(?P<version>[^-]+)"
    r"(?:-(?P<build>\d[^-]*))?"
    r"-(?P<python>[^-]+)-(?P<abi>[^-]+)-(?P<platform>[^-]+)\.whl$"
)


def parse_wheel(filename):
    match = WHEEL_RE.match(filename)
    if match is None:
        raise ValueError(f"not a valid wheel filename: {filename}")
    return match


def normalize(name):
    """PEP 503 normalized project name, as used in the simple index URL."""
    return re.sub(r"[-_.]+", "-", name).lower()


def local_wheels(directory):
    """Project name and version shared by every wheel in *directory*."""
    names = sorted(f for f in os.listdir(directory) if f.endswith(".whl"))
    if not names:
        raise SystemExit(f"no wheels found in {directory}")

    parsed = {(m["name"], m["version"]) for m in map(parse_wheel, names)}
    if len(parsed) != 1:
        raise SystemExit(f"wheels disagree on name/version: {sorted(parsed)}")

    (name, version), = parsed
    return name, version, len(names)


def published_build_tags(index_url, name, version):
    """Build tags already used on the index for this exact version.

    An unknown project, or one whose files have all been deleted, yields an
    empty set. Note that deleted filenames stay unusable even though they no
    longer appear here, which is why --build-tag exists as an escape hatch.
    """
    # The index sits behind a CDN that can still be serving the previous
    # listing moments after an upload. A stale answer here would hand back a
    # build tag that is already taken, so ask for an uncached copy.
    url = (
        f"{index_url.rstrip('/')}/{normalize(name)}/"
        f"?cache_bust={uuid.uuid4().hex}"
    )
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.pypi.simple.v1+json",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return set()
        raise

    tags = set()
    for entry in payload.get("files", []):
        filename = entry.get("filename", "")
        if not filename.endswith(".whl"):
            continue
        try:
            match = parse_wheel(filename)
        except ValueError:
            continue
        if match["version"] != version:
            continue
        # A file published without a build tag occupies the untagged
        # filename, so the first tagged upload can still be 1.
        if match["build"] is not None:
            tags.add(match["build"])

    return tags


BUILD_TAG_RE = re.compile(r"^\d[\w.]*$")


def next_build_tag(tags):
    """Successor of the highest purely numeric tag, or 1 if there is none."""
    numbers = [int(tag) for tag in tags if tag.isdigit()]
    return max(numbers) + 1 if numbers else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="directory holding the built wheels")
    parser.add_argument(
        "--index-url",
        default="https://test.pypi.org/simple",
        help="simple index to query (default: %(default)s)",
    )
    parser.add_argument(
        "--build-tag",
        default=os.environ.get("BUILD_TAG_OVERRIDE", ""),
        help="skip the index query and use this build tag instead",
    )
    args = parser.parse_args(argv)

    name, version, count = local_wheels(args.directory)
    print(f"{count} wheels, {name} {version}", file=sys.stderr)

    override = args.build_tag.strip()
    if override:
        # A build tag starts with a digit (PEP 427). Checking here keeps a
        # malformed workflow input from reaching the shell or GITHUB_OUTPUT.
        if not BUILD_TAG_RE.match(override):
            raise SystemExit(f"not a valid build tag: {override!r}")
        build_tag = override
        print(f"build tag {build_tag} (override)", file=sys.stderr)
    else:
        tags = published_build_tags(args.index_url, name, version)
        build_tag = next_build_tag(tags)
        published = ", ".join(sorted(tags)) if tags else "none"
        print(f"published build tags: {published}", file=sys.stderr)
        print(f"build tag {build_tag}", file=sys.stderr)

    print(f"version={version}")
    print(f"build_tag={build_tag}")


if __name__ == "__main__":
    main()
