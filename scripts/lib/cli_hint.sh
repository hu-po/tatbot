#!/usr/bin/env bash
# One line on stderr when a launcher is run by path instead of through the
# `tatbot` CLI (docs/cli.md). SOURCE this file, then:
#
#   cli_hint::note "tatbot record"
#
# Silent when the CLI is the caller (it exports TATBOT_VIA_CLI=1) or when the
# operator sets TATBOT_NO_HINT=1. It never changes behaviour or exit status —
# the launchers keep working by path; the hint is how they teach the verb.
#
# shellcheck shell=bash

cli_hint::note() {
  [ "${TATBOT_VIA_CLI:-0}" = 1 ] && return 0
  [ "${TATBOT_NO_HINT:-0}" = 1 ] && return 0
  printf '\033[2m→ this is `%s` in the tatbot CLI (scripts/tatbot; TATBOT_NO_HINT=1 silences this)\033[0m\n' "$1" >&2
  return 0
}
