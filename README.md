# SAF → Alpamayo 확장 실험 진행 기록 (2026-08-14)

## 0. 목표

기존 SAF(Semantic-Aware Flushing, Moondream2 VLA + Jetson Orin NX 기반)를 **Alpamayo**(NVIDIA의 자율주행용 reasoning VLA, 10B)로 확장. 오늘은 환경 구축 + 방향 탐색까지 진행.

---

## 1. 서버/환경 정보

| 항목 | 값 |
|---|---|
| 서버 | 학과 서버, A100 80GB ×3 (스케줄러 없음, SSH 직접 공유) |
| **사용 GPU** | **GPU 1번** (비어있음 — GPU 0은 남이 소량 사용 중, GPU 2는 다른 사람이 85% 점유 중이라 절대 사용 금지) |
| 작업 디렉토리 | `/mnt/nvme/bigsyslab/alpamayo_saf/alpamayo` (NVMe, root 파티션은 91% 차 있어서 회피) |
| HF/uv 캐시 | `/mnt/nvme/bigsyslab/alpamayo_saf/.hf_cache`, `.uv_cache` (환경변수로 지정, `~/.bashrc`에 등록됨) |
| CUDA 툴킷 | `/usr/local/cuda-12.0` (시스템 기본 `/usr/bin/nvcc`는 11.5라 사용 불가; 12.4는 nvcc 자체가 없는 부분설치였음) |
| conda env | `alpamayo` (Python 3.12) |
| 실제 패키지 위치 | conda env가 아니라 **프로젝트 내 `.venv`** (`uv sync --active`가 만듦) — 매번 별도 활성화 필요 |

### 매번 접속 시 활성화 순서
```bash
conda activate alpamayo
cd /mnt/nvme/bigsyslab/alpamayo_saf/alpamayo
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
```

---

## 2. 환경 구축 중 겪은 이슈 (다음엔 안 겪어도 됨)

1. **`flash-attn` 빌드 실패**: 시스템 기본 `nvcc`가 11.5라 (flash-attn은 11.7+ 요구) → `/usr/local/cuda-12.0`으로 PATH/CUDA_HOME 재지정 후 해결.
2. **HF gated 관련**: 모델(`nvidia/Alpamayo-R1-10B`)은 실제론 **gated 아님** (공개). 다만 예제 클립이 쓰는 **데이터셋** `nvidia/PhysicalAI-Autonomous-Vehicles`는 gated — HF 사이트에서 라이선스 클릭 동의 필요(승인 대기 없이 즉시 열림).
3. **heredoc paste 문제**: tmux + 긴 heredoc 붙여넣기가 여러 번 깨짐 → **로컬 Mac에서 `scp`로 파일 전송**하는 게 제일 안정적.

---

## 3. 베이스라인 검증 결과 (`test_inference.py`)

| 실행 | 상태 | Elapsed | 비고 |
|---|---|---|---|
| 1차 (콜드, 다운로드 포함) | 성공 | 4:17 | `Fetching 5 files` 다운로드 2:50 포함 |
| 2차 (캐시됨) | 성공 | 1:07.91 | 순수 로딩+추론 |

- **`Major page faults: 0`** — 두 실행 모두 디스크에서 실제로 읽어온 적이 없음 (체크포인트가 OS 페이지캐시에 상주). 즉 **웜 캐시 상태에서 Alpamayo 추론은 스토리지와 거의 무관 (GPU compute-bound, CPU 사용률 36%)**.
- `iostat` 관찰 중 06:00:45~06:03:42 구간에 GB/s급 쓰기 폭주 발견 → `pidstat`으로 재확인했으나 **시간대가 안 맞아 원인 미확정** (다른 사용자 프로세스일 가능성 높음, 공유 서버라는 방증). 우선순위 낮아서 보류.

---

## 4. 방향 탐색 — 실배포 구조 조사 결과

- **Alpamayo(10B~34B) 자체는 차량에 안 올라감.** 클라우드/온프렘의 "teacher 모델"로, reasoning trace/trajectory를 생성해 라벨링 → 이를 **distill/양자화한 student 모델이 차량 내 DRIVE AGX Thor**(128GB 통합메모리, INT4/FP4)에서 10Hz 실시간 구동.
- **차량 엣지에서는 스토리지 오프로딩 필요성이 낮음** (Thor의 128GB가 애초에 오프로드 안 하도록 설계됨). → "엣지에서 SAF"라는 원래 프레이밍은 실제 배포와 안 맞음.
- **스토리지 관련성은 클라우드 측(teacher 모델) 동시 서빙 상황**에서 성립 가능 — 여러 요청 동시 처리 시 VRAM 압박 → KV캐시/가중치 오프로드 필요해질 수 있음.

## 5. 교수님 제안 방향: 가중치 hot/cold 오프로딩

- 자주 쓰는 가중치는 VRAM/DRAM에, 나머지는 스토리지로 오프로딩하자는 아이디어.
- **선행연구 매우 많음**: FlexGen, "LLM in a Flash"(Apple, 이미 relatedWorks 폴더에 있음), PowerInfer, DeepSpeed ZeRO-Infinity, DUAL-BLADE(NVMe KV 오프로딩), NVIDIA ICMSP/NIXL(2026 CES, 인프라 레벨 KV 오프로딩), MoE expert 오프로딩 계열(MoE-Infinity, FineMoE, **VisMMOE — 2026.05, Vision-Language MoE 오프로딩으로 사실상 동일 아이디어 선점**).
- **차별화 각도로 합의한 방향**:
  1. Alpamayo의 **이중 서브모듈 구조**(아래 6번 참고)의 비대칭적 접근 패턴 활용 — 기존 문헌엔 없는 VLA 고유 구조
  2. 기존 문헌은 대부분 처리량/비용 최적화 프레임 → 우리는 **자율주행 실시간 tail latency/jitter 보장** 프레임으로 차별화
  3. SAF의 핵심 IP(semantic boundary 기반 I/O 타이밍)를 **가중치 prefetch**에 재적용 — reasoning 단계→trajectory decode 전환 시점을 prefetch 트리거로 사용

---

## 6. 오늘의 핵심 실측 결과 — 아키텍처 확인 (`inspect_alpamayo_arch.py`)

```
전체 파라미터: 11.08B

vlm             | Qwen3VLForConditionalGeneration | 8.80B params   ← reasoning 백본
expert          | Qwen3VLTextModel                | 2.28B params   ← action/trajectory expert
action_space    | UnicycleAccelCurvatureActionSpace | ~0B
diffusion       | FlowMatching                     | ~0B (wrapper, 실제 연산은 expert가 담당)
action_in_proj  | PerWaypointActionInProjV2        | ~0B
action_out_proj | Linear                           | ~0B
```

- **가설대로 두 서브모듈이 완전히 분리돼 있음이 확인됨** → 이후 오프로딩 코드에서 `model.vlm` / `model.expert`를 바로 대상으로 삼으면 됨.
- **MoE 아님, dense 확정.** (검색 스크립트가 "expert"라는 서브모듈 이름 때문에 544건 오탐 — 실제 라우팅 구조 없음, 그냥 이름이 "action expert"인 두 번째 dense 트랜스포머)
- 둘 다 **Qwen3VL 계열** 아키텍처 기반 (Cosmos-Reason은 브랜드명, 실구현은 Qwen3VL).

### 미완료: 서브모듈별 호출 횟수/시간 프로파일링
`sample_trajectories_from_data_with_vlm_rollout` 호출 시 `KeyError: 'tokenized_data'` 발생 — `load_physical_aiavdataset()`로 받은 raw data를 모델에 넣기 전에 **중간 전처리/토크나이징 단계**가 필요한데, 이 부분을 실제 `test_inference.py` 소스를 안 보고 짐작해서 코드를 짜서 빠뜨림. **다음 세션에서 `test_inference.py` 전체를 먼저 읽고 정확한 전처리 파이프라인을 반영해서 재시도 필요.**

---

## 7. 다음 세션 할 일 (우선순위 순)

1. `src/alpamayo_r1/test_inference.py` 전체 소스 읽고, `data`를 `sample_trajectories_from_data_with_vlm_rollout`에 넣기 전 정확한 전처리 단계 파악
2. `inspect_alpamayo_arch.py`의 forward hook 프로파일링 재실행 → `vlm` vs `expert` 호출 횟수/누적시간 실측 (이게 hot/cold 우선순위 결정 근거)
3. 위 결과 바탕으로 오프로딩 실험 하네스 설계 시작:
   - `full_resident` / `naive_offload` / `static_hotcold` / `saf_prefetch` 4개 조건
   - 지표: 기존 SAF 지표(mean/P50/P95/P99/IQR/CV%) + **weight stall time** 신규 추가
   - A100 80GB에선 자연 메모리 압박이 없으므로, `torch.cuda.set_per_process_memory_fraction()` 등으로 VRAM을 인위적으로 제한해서 오프로드가 실제로 필요한 상황을 재현해야 함

---

## 8. 파일 위치 메모

- 로컬(제 스크래치패드): `inspect_alpamayo_arch.py` 원본 — `/private/tmp/claude-502/.../scratchpad/inspect_alpamayo_arch.py`
- 서버: `/mnt/nvme/bigsyslab/alpamayo_saf/alpamayo/inspect_alpamayo_arch.py`
- 서버 로그: `/tmp/run_alpamayo_probe1.log`, `probe2.log`, `/tmp/iostat_probe1.log`, `/tmp/pidstat_probe.log`, `/tmp/arch_probe.log`
