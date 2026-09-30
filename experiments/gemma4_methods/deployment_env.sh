#!/usr/bin/env bash
# Source-only deployment gate. No fallback runtime, overlay, or submission.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Source deployment_env.sh from a reviewed launcher.' >&2
    exit 2
fi
: "${GEMMA_DEPLOY_REPO:?Exact canonical deployed checkout required}"
: "${GEMMA_DEPLOY_COMMIT:?Exact incorporated source commit required}"
: "${GEMMA_RUNTIME_PYTHON:?Pinned runtime interpreter required}"
: "${GEMMA_RUNTIME_MANIFEST:?Runtime pins manifest required}"
: "${GEMMA_RUNTIME_MANIFEST_SHA256:?Reviewed runtime manifest digest required}"
case "$GEMMA_DEPLOY_COMMIT" in
    *[!0-9a-f]*|'') echo 'Invalid deployment commit.' >&2; return 2;;
esac
[[ ${#GEMMA_DEPLOY_COMMIT} == 40 ]] || return 2
[[ "$GEMMA_DEPLOY_REPO" == /* && "$GEMMA_RUNTIME_PYTHON" == /* ]] || return 2
gemma_launch_profile="${GEMMA_LAUNCH_PROFILE:-generation}"
case "$gemma_launch_profile" in generation|grading) ;; *) return 2;; esac
if [[ "$gemma_launch_profile" == generation ]]; then
: "${GEMMA_RUNTIME_ENV:?Tracked relative runtime environment script required}"
case "$GEMMA_RUNTIME_ENV" in
    /*|../*|*/../*|*/..|'') echo 'Runtime environment must be repository-relative.' >&2; return 2;;
esac
fi
gemma_deploy_root=$(cd "$GEMMA_DEPLOY_REPO" && pwd -P)
[[ "$(git -C "$gemma_deploy_root" rev-parse --show-toplevel)" == "$gemma_deploy_root" ]] || return 2
[[ "$(git -C "$gemma_deploy_root" rev-parse HEAD)" == "$GEMMA_DEPLOY_COMMIT" ]] || {
    echo 'Deployment HEAD differs from the approved exact commit.' >&2; return 2;
}
[[ -z "$(git -C "$gemma_deploy_root" status --porcelain --untracked-files=all)" ]] || {
    echo 'Deployment source must be clean, including untracked files.' >&2; return 2;
}
git -C "$gemma_deploy_root" merge-base --is-ancestor \
    45f27c24855c82f3dc81019bd246a6f7641a13ec "$GEMMA_DEPLOY_COMMIT" || return 2
[[ -x "$GEMMA_RUNTIME_PYTHON" ]] || return 2
cd "$gemma_deploy_root"
export CTM_MUSE_PYTHON="$GEMMA_RUNTIME_PYTHON"
export PYTHONPATH="$gemma_deploy_root" PYTHONNOUSERSITE=1
if [[ "$gemma_launch_profile" == generation ]]; then
    git -C "$gemma_deploy_root" ls-files --error-unmatch -- "$GEMMA_RUNTIME_ENV" >/dev/null || return 2
    [[ ! -L "$gemma_deploy_root/$GEMMA_RUNTIME_ENV" ]] || return 2
    source "$gemma_deploy_root/$GEMMA_RUNTIME_ENV" || return 2
else
    export CTM_MUSE_RUNTIME_PYTHON="$GEMMA_RUNTIME_PYTHON"
fi
[[ "$CTM_MUSE_RUNTIME_PYTHON" == "$GEMMA_RUNTIME_PYTHON" ]] || {
    echo 'Runtime setup substituted a different interpreter.' >&2; return 2;
}
# Remove any inherited or runtime-added repository/client overlay.
export PYTHONPATH="$gemma_deploy_root" PYTHONNOUSERSITE=1
gemma_guard_factory=()
if [[ -n "${GEMMA_VERIFIER_FACTORY:-}" ]]; then
    gemma_guard_factory=(--verifier-factory "$GEMMA_VERIFIER_FACTORY")
fi
"$GEMMA_RUNTIME_PYTHON" -B -m experiments.gemma4_methods.launch_guard \
    --repository "$gemma_deploy_root" --commit "$GEMMA_DEPLOY_COMMIT" \
    --manifest "$GEMMA_RUNTIME_MANIFEST" --manifest-sha256 "$GEMMA_RUNTIME_MANIFEST_SHA256" \
    --profile "$gemma_launch_profile" \
    "${gemma_guard_factory[@]}" || return 2
unset gemma_deploy_root gemma_guard_factory gemma_launch_profile
