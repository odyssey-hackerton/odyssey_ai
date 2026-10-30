#!/usr/bin/env bash
# GitHub 저장소 설정 + main/dev 브랜치 보호(Ruleset)를 적용한다.
# 필요: gh CLI 로그인(gh auth login), 저장소 admin 권한, 원격에 main·dev 브랜치가 있어야 함
#
# 사용법: ./scripts/setup-branch-protection.sh [owner/repo]
#   승인 인원 변경: APPROVALS_MAIN=1 APPROVALS_DEV=0 ./scripts/setup-branch-protection.sh
set -euo pipefail

REPO="${1:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}"
APPROVALS_MAIN="${APPROVALS_MAIN:-1}"
APPROVALS_DEV="${APPROVALS_DEV:-1}"
CHECK_NAME="convention-check"

echo "▶ 대상 저장소: $REPO"

for b in main dev; do
  if ! gh api "repos/$REPO/branches/$b" >/dev/null 2>&1; then
    echo "✖ 원격에 '$b' 브랜치가 없습니다. 먼저 push 하세요." >&2
    exit 1
  fi
done

echo "▶ 저장소 설정: 기본 브랜치 dev, 병합 후 브랜치 자동 삭제, rebase merge 비활성화"
gh api -X PATCH "repos/$REPO" \
  -f default_branch=dev \
  -F delete_branch_on_merge=true \
  -F allow_squash_merge=true \
  -F allow_merge_commit=true \
  -F allow_rebase_merge=false >/dev/null

apply_ruleset() {
  local name="$1" branch="$2" approvals="$3" merge_methods="$4"
  local body
  body=$(cat <<JSON
{
  "name": "$name",
  "target": "branch",
  "enforcement": "active",
  "conditions": { "ref_name": { "include": ["refs/heads/$branch"], "exclude": [] } },
  "bypass_actors": [],
  "rules": [
    { "type": "deletion" },
    { "type": "non_fast_forward" },
    {
      "type": "pull_request",
      "parameters": {
        "required_approving_review_count": $approvals,
        "dismiss_stale_reviews_on_push": true,
        "require_code_owner_review": false,
        "require_last_push_approval": false,
        "required_review_thread_resolution": true,
        "allowed_merge_methods": $merge_methods
      }
    },
    {
      "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": false,
        "required_status_checks": [ { "context": "$CHECK_NAME" } ]
      }
    }
  ]
}
JSON
)
  local id
  id=$(gh api "repos/$REPO/rulesets" --jq ".[] | select(.name==\"$name\") | .id" || true)
  if [ -n "$id" ]; then
    echo "$body" | gh api -X PUT "repos/$REPO/rulesets/$id" --input - >/dev/null
    echo "✔ ruleset 갱신: $name"
  else
    echo "$body" | gh api -X POST "repos/$REPO/rulesets" --input - >/dev/null
    echo "✔ ruleset 생성: $name"
  fi
}

apply_ruleset "protect-main" main "$APPROVALS_MAIN" '["merge"]'
apply_ruleset "protect-dev"  dev  "$APPROVALS_DEV"  '["squash", "merge"]'

echo "완료. GitHub > Settings > Rules > Rulesets 에서 확인하세요."
