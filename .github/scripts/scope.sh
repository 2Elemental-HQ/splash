#!/bin/bash
# Prints a cache key that names the content of the given paths at HEAD:  scope.sh <prefix> <path>...
#
# A job that passed for exactly this content leaves a marker under this key (see ci.yml and splash-manager.yml);
# a later run with the same key skips the job. The key is a hash of the git blobs, so it follows the content,
# not the commit: an app-only commit leaves the keys of the engine and the Python server unchanged, while any
# change to what a job covers, or to its workflow, makes a new key and the job runs. Missing paths are ignored.
set -euo pipefail
prefix=$1; shift
printf '%s-%s\n' "$prefix" "$(git ls-tree -r HEAD -- "$@" | sha256sum | cut -c1-40)"
