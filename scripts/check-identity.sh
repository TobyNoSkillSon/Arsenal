#!/bin/bash
# Check, before a build depends on it, that codesign can use "Arsenal Release Signing" without a keychain prompt.
# Replaces make-identity.sh: the identity is never created here and never lives in the login keychain. Locally run the
# build through with-release-keychain.sh (private work area); in CI, ci-keychain.sh import. Both set
# ARSENAL_SIGN_IDENTITY and ARSENAL_SIGN_KEYCHAIN.
set -euo pipefail
source "$(dirname "$0")/pkg-lib.sh"
IDENTITY="$(resolve_identity)"
security find-identity -p codesigning ${ARSENAL_SIGN_KEYCHAIN:+"$ARSENAL_SIGN_KEYCHAIN"} | grep -Fq -- "$IDENTITY" \
  || die "$SIGN_NAME ($IDENTITY) is not available to codesign. Local builds: with-release-keychain.sh scripts/build.sh; CI: scripts/ci-keychain.sh import"
probe="$(mktemp "${TMPDIR:-/tmp}/arsenal-probe.XXXXXX")"
trap 'rm -f "$probe"' EXIT
cp /usr/bin/true "$probe"
perl -e 'alarm 30; exec @ARGV' codesign ${ARSENAL_SIGN_KEYCHAIN:+--keychain "$ARSENAL_SIGN_KEYCHAIN"} --force --timestamp=none --sign "$IDENTITY" "$probe" 2>/dev/null \
  || die "codesign cannot use $SIGN_NAME without a prompt; the temporary keychain needs its partition list (with-release-keychain.sh sets it)"
echo "identity: $SIGN_NAME $IDENTITY"
