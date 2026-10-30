#!/usr/bin/env bash
# 로컬(Mac)에서 실행: GPU 서버용(linux/amd64) 워커 이미지를 빌드해 Docker Hub에 올린다.
#
# 사용법:
#   docker login
#   DOCKERHUB_USER=<아이디> ./scripts/build-push.sh          태그 = git 커밋 해시
#   DOCKERHUB_USER=<아이디> ./scripts/build-push.sh v0.1.0   태그 직접 지정
#
# Apple Silicon Mac에서 그냥 빌드하면 arm64 이미지가 되어 x86 서버에서 실행되지 않으므로
# 플랫폼을 linux/amd64로 고정한다.
set -euo pipefail

: "${DOCKERHUB_USER:?DOCKERHUB_USER 를 지정하세요 (예: DOCKERHUB_USER=myid ./scripts/build-push.sh)}"
PLATFORM="${PLATFORM:-linux/amd64}"

cd "$(git rev-parse --show-toplevel)"

TAG="${1:-$(git rev-parse --short HEAD)}"
if [ -z "${1:-}" ] && [ -n "$(git status --porcelain)" ]; then
  TAG="$TAG-dirty"   # 커밋 안 된 변경이 들어간 이미지임을 표시
fi

IMAGE="$DOCKERHUB_USER/odyssey-ai"
echo "▶ $IMAGE:$TAG ($PLATFORM)"
docker buildx build --platform "$PLATFORM" \
  -t "$IMAGE:$TAG" -t "$IMAGE:latest" \
  --push .

echo
echo "✔ 푸시 완료: $IMAGE:$TAG (latest도 함께 갱신)"
echo "  GPU 서버에서: docker pull $IMAGE:latest 후 README 의 실행 명령"
