# odyssey_ai

GPU 서버에서 도는 AI 워커입니다. 일반 서버의 Redis에서 작업을 가져와 ① 탐지 → ② 위험 분석 → ③ 변환 → ② 재검증(최대 3회)을 거쳐 결과를 돌려보냅니다. 협업 규칙은 [CONTRIBUTING.md](CONTRIBUTING.md)를 보세요.

지금은 AI 모델이 없어서 **아무것도 찾지 않고 사진·글을 그대로 돌려줍니다**. 큐 연결, 진행 상태, S3, 결과 전송은 실제와 똑같이 동작하므로 AI 팀은 `stages/`만 채우면 됩니다.

```
일반 서버 (10.1.1.27)                           GPU 서버
API ──XADD──▶ Redis jobs ──XREADGROUP──▶ worker.main ──▶ pipeline ──▶ stages (detect, risk, transform)
API ◀──────── Redis results ◀──XADD──── worker.main
              Redis job:{id} ◀── 진행 상태 (프론트 진행 표시)
              S3 originals/ ──▶ 원본 사진 / 결과 사진 ──▶ S3 results/
```

GPU 서버는 Redis(일반 서버 사설 IP, 6379)와 S3에만 접속합니다. Postgres에는 접속하지 않고, 들어오는 포트도 열 필요가 없습니다.

## 폴더 구조

```
stages/          ← AI 팀 작업 영역
  detect.py      ① 탐지 (PaddleOCR, 얼굴, EXIF, 글)
  risk.py        ② 위험 분석·재검증
  transform.py   ③ 변환 (블러, 마스킹, EXIF 제거)
  types.py       Media(사진·글), Assessment(위험도·탐지 항목)
worker/
  main.py        작업 받기 → 처리 → 결과 보내기 (수정할 일 거의 없음)
  pipeline.py    단계 순서와 반복 규칙
  contract.py    백엔드와 주고받는 메시지 형식 (백엔드와 같게 유지)
  check.py       통신 확인 (Redis, S3)
  queue.py       Redis 연결·진행 상태·결과 전송
  storage.py     S3 읽기·쓰기
```

입력 종류(`input_type`)는 세 가지이고 `Media`에 그대로 들어옵니다.

| input_type | Media.image | Media.text |
| --- | --- | --- |
| `image` | 있음 | 없음 |
| `text` | 없음 | 있음 |
| `image_text` | 있음 | 있음 (사진과 함께 올린 글. 같이 분석) |

## AI 넣는 방법

1. `stages/detect.py`, `risk.py`, `transform.py`의 `TODO(AI)` 부분을 채웁니다. 함수의 입력·출력 형식만 지키면 내부는 자유입니다.
2. 탐지 항목(`findings`)은 `type`, `source`, `bbox`, `score`, `action` 키를 쓰고, 필요하면 다른 키를 더 넣어도 됩니다. 백엔드가 그대로 저장해서 리포트에 보여줍니다.
3. 모델 라이브러리를 추가했으면 `uv add <패키지>`로 `pyproject.toml`과 `uv.lock`을 함께 갱신합니다.
4. GPU를 쓰게 되면 `Dockerfile`의 베이스 이미지를 CUDA 이미지로 바꾸고, 실행할 때 `--gpus all`을 붙입니다.

## 로컬 개발

```bash
uv sync
cp .env.example .env
uv run python -m worker.check
uv run python -m worker.main
uv run pytest
uv run ruff check . && uv run ruff format .
```

`.env`의 `REDIS_URL`은 VPN에 연결된 상태에서 일반 서버 Redis를 가리키거나, 로컬 Redis를 씁니다. 로컬에서 끝까지 확인하려면 백엔드 레포를 같이 띄웁니다.

## GPU 서버 배포

### 1. 이미지 빌드·푸시 (Mac)

```bash
docker login
DOCKERHUB_USER=jeongjaeyoon ./scripts/build-push.sh
```

### 2. 통신 확인 (GPU 서버)

비밀번호를 명령 기록에 남기지 않도록 입력받아서 씁니다. Redis 비밀번호는 일반 서버 `.env`의 `REDIS_PASSWORD`입니다.

```bash
read -s -p "Redis 비밀번호: " RP; echo
docker run --rm -e REDIS_URL="redis://:${RP}@10.1.1.27:6379/0" jeongjaeyoon/odyssey-ai:latest python -m worker.check
```

`✔ Redis 접속·인증`이 나오면 통신은 정상입니다. S3 키를 넣지 않았으면 S3는 건너뜁니다.

### 3. 워커 실행 (GPU 서버)

```bash
read -s -p "Redis 비밀번호: " RP; echo
docker run -d --name odyssey-worker --restart unless-stopped -e REDIS_URL="redis://:${RP}@10.1.1.27:6379/0" jeongjaeyoon/odyssey-ai:latest
unset RP
docker logs -f odyssey-worker
```

사진 작업까지 처리하려면 워커용 AWS 키를 함께 넣습니다 (`-e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=...`, 권한은 `originals/*` 읽기와 `results/*` 쓰기).

### 4. 업데이트·확인

```bash
docker pull jeongjaeyoon/odyssey-ai:latest
docker rm -f odyssey-worker
```

그다음 3번을 다시 실행합니다.

- 일반 서버 Mock 워커는 꺼 둡니다 (`.env`의 `COMPOSE_PROFILES=mock` 주석 처리 후 `docker compose stop mock-worker`). 켜 두면 작업이 두 워커에 나뉘어 갑니다.
- 일반 서버나 Redis가 재시작되어도 워커는 죽지 않고 5초마다 다시 붙습니다.
- 처리 도중 워커가 죽으면 그 작업은 10분 뒤 다시 가져와 처리합니다.
