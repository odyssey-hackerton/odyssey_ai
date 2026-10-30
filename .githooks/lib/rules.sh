#!/usr/bin/env bash
# 브랜치·커밋 규칙 정의. CI(.github/workflows/convention-check.yml)와 동일하게 유지할 것.

PROTECTED_BRANCHES="main dev"
BRANCH_WORK_RE='^(feature|fix|refactor|docs|chore|test|ci)/([0-9]+-)?[a-z0-9]+(-[a-z0-9]+)*$'
BRANCH_HOTFIX_RE='^hotfix/([0-9]+-)?[a-z0-9]+(-[a-z0-9]+)*$'
BRANCH_RELEASE_RE='^release/v[0-9]+\.[0-9]+\.[0-9]+$'

COMMIT_TYPES='feat|fix|refactor|docs|style|test|chore|perf|ci|build|revert'
COMMIT_HEADER_RE="^(${COMMIT_TYPES})(\([a-z0-9-]+\))?!?: [^ ].*$"
COMMIT_HEADER_MAX=72
# 자동 생성 메시지는 검사 제외
COMMIT_SKIP_RE='^(Merge |Revert "|fixup! |squash! |amend! )'

is_protected_branch() {
  local b
  for b in $PROTECTED_BRANCHES; do
    [ "$1" = "$b" ] && return 0
  done
  return 1
}

is_valid_branch() {
  is_protected_branch "$1" && return 0
  printf '%s\n' "$1" | grep -Eq "${BRANCH_WORK_RE}|${BRANCH_HOTFIX_RE}|${BRANCH_RELEASE_RE}"
}

char_length() {
  # 한글을 바이트가 아닌 글자 수로 세기
  if command -v python3 >/dev/null 2>&1; then
    printf '%s' "$1" | python3 -c 'import sys; print(len(sys.stdin.read()))'
  else
    printf '%s' "$1" | wc -m | tr -d ' '
  fi
}

print_branch_help() {
  cat >&2 <<'HELP'
  브랜치 이름 규칙: <type>/<이슈번호>-<짧은-설명>   (이슈번호 선택, 소문자·숫자·하이픈만)
    type: feature fix refactor docs chore test ci hotfix
    release는 release/v1.0.0 형식
  예) feature/12-upload-api, fix/sse-reconnect, hotfix/minio-timeout
  자세한 내용: CONTRIBUTING.md
HELP
}
