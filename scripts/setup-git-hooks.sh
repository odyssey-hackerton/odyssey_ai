#!/usr/bin/env bash
# 클론 후 1회 실행: 검사 훅과 커밋 템플릿을 이 레포에 연결한다.
set -e
cd "$(git rev-parse --show-toplevel)"

chmod +x .githooks/pre-commit .githooks/commit-msg .githooks/pre-push .githooks/reference-transaction
git config core.hooksPath .githooks
git config commit.template .gitmessage.txt

ver=$(git --version | sed -E 's/[^0-9]*([0-9]+)\.([0-9]+).*/\1 \2/')
set -- $ver
if [ "$1" -lt 2 ] || { [ "$1" -eq 2 ] && [ "$2" -lt 28 ]; }; then
  echo "⚠ git $1.$2 에서는 브랜치 생성 검사(reference-transaction)가 동작하지 않습니다. git 2.28 이상을 권장합니다."
fi

echo "✔ 설정 완료"
echo "  - core.hooksPath = .githooks (브랜치 생성, 브랜치·커밋 메시지, push 검사)"
echo "  - commit.template = .gitmessage.txt"
