# 협업 규칙

처음 클론했다면 아래 명령을 한 번 실행하세요. 브랜치 이름·커밋 메시지 검사 훅과 커밋 템플릿이 설정됩니다.

```bash
./scripts/setup-git-hooks.sh
```

## 1. 브랜치 전략

```
main  ← 배포(시연) 브랜치. 직접 push 금지, PR로만 병합
 └─ dev  ← 통합 브랜치. 직접 push 금지, PR로만 병합
     └─ feature/*, fix/* ...  ← 작업 브랜치. dev에서 따서 dev로 PR
```

| 브랜치 | 만드는 곳 | PR 대상 | 병합 방식 |
|---|---|---|---|
| `feature/*` `fix/*` `refactor/*` `docs/*` `chore/*` `test/*` `ci/*` | `dev` | `dev` | Squash merge |
| `release/*` | `dev` | `main` (병합 후 `dev`에도 반영) | Merge commit |
| `hotfix/*` | `main` | `main` (병합 후 `dev`에도 반영) | Merge commit |
| `dev` | - | `main` | Merge commit |

### 브랜치 이름 규칙

```
<type>/<이슈번호>-<짧은-설명>     이슈번호는 선택
```

- 소문자, 숫자, 하이픈(`-`)만 사용 (kebab-case)
- type: `feature` `fix` `refactor` `docs` `chore` `test` `ci` `hotfix`
- release는 버전 형식: `release/v1.0.0`

| 올바른 예 | 잘못된 예 | 이유 |
|---|---|---|
| `feature/12-upload-api` | `feature/UploadAPI` | 대문자 |
| `fix/sse-reconnect` | `fix/sse_reconnect` | 언더스코어 |
| `hotfix/minio-timeout` | `feat/upload` | type은 `feature` |
| `release/v1.0.0` | `upload-api` | type 없음 |

규칙에 맞지 않는 이름은 로컬에서 **브랜치를 만드는 순간 거부**되고(훅 설치 시), GitHub에서도 CI가 실패합니다.
이미 만든 브랜치는 `git branch -m <새-이름>`으로 바꾸면 됩니다.

## 2. 커밋 메시지 규칙 (Conventional Commits)

```
<type>(<scope>): <subject>

<body>

<footer>
```

- header는 72자 이내, 끝에 마침표 없음, 한국어 가능
- scope는 선택, 소문자·숫자·하이픈
- body: 무엇을, 왜 바꿨는지
- footer: `Closes #12`, `BREAKING CHANGE: ...`

| type | 용도 |
|---|---|
| `feat` | 새 기능 |
| `fix` | 버그 수정 |
| `refactor` | 동작 변화 없는 구조 개선 |
| `docs` | 문서 |
| `style` | 포맷, 세미콜론 등 (로직 변화 없음) |
| `test` | 테스트 추가·수정 |
| `chore` | 빌드 설정, 패키지 등 기타 |
| `perf` | 성능 개선 |
| `ci` | CI 설정 |
| `build` | 빌드 시스템, 의존성 |
| `revert` | 커밋 되돌리기 |

scope 예시
- backend: `api` `upload` `job` `sse` `db` `storage` `queue` `infra`
- ai: `detect` `ocr` `risk` `transform` `verify` `worker` `queue` `model`

```
feat(upload): 이미지 업로드 API 추가
fix(ocr): 회전된 이미지에서 좌표가 어긋나는 문제 수정
chore: Dockerfile 베이스 이미지 변경
```

## 3. Pull Request 규칙

- PR 제목은 커밋 메시지 header 규칙을 따릅니다. (dev로 squash merge 시 PR 제목이 그대로 커밋이 됨)
- 대상 브랜치
  - 작업 브랜치 → `dev`
  - `dev`, `release/*`, `hotfix/*` → `main`
- 리뷰 승인 1명 이상, `convention-check` 통과 후 병합
- 병합된 작업 브랜치는 삭제합니다.
- 키, 비밀번호, `.env`, 실제 사용자 사진은 절대 커밋하지 않습니다.

## 4. 작업 흐름 예시

```bash
git switch dev && git pull
git switch -c feature/12-upload-api
# 작업 후
git add . && git commit        # 템플릿이 열림
git push -u origin feature/12-upload-api
# GitHub에서 dev 대상으로 PR 생성
```

## 5. 검사 장치 요약

| 위치 | 검사 내용 |
|---|---|
| 로컬 `reference-transaction` 훅 | 규칙에 안 맞는 브랜치 생성 차단 |
| 로컬 `pre-commit` 훅 | 규칙에 안 맞는 브랜치에서 커밋 차단 (`git branch -m`으로 이름을 바꾼 경우 대비) |
| 로컬 `commit-msg` 훅 | 커밋 메시지 형식 검사 |
| 로컬 `pre-push` 훅 | `main`/`dev` 직접 push 차단, 원격 브랜치 이름 검사 |
| GitHub Actions `convention-check` | 브랜치 이름, PR 대상 브랜치, PR 제목, PR 내 모든 커밋 메시지 |
| GitHub Ruleset | `main`/`dev` 직접 push·강제 push·삭제 금지, PR·승인·CI 통과 필수 |
