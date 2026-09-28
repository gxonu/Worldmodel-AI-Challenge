# INHA AI Challenge 재현 코드 - 팀: 최후의 18시간

## 1. 실행 전 반드시 채워야 하는 폴더

> **중요:** ZIP에는 `data/`, `submission_kit/`, `pretrained/`의 빈 디렉터리만 포함되어 있습니다. 
> 이 세 폴더는 용량과 원본 재배포 문제로 내용물을 포함하지 않았습니다. 
> 아래 구조대로 대회 `open.zip` 원본과 pretrained Cosmos-Predict2.5-2B를 반드시 채운 뒤 실행해야 합니다.
> 채우는 방법은 아래 명시되어 있습니다. 
> 파일이 없거나 경로가 다르면 학습/추론이 즉시 중단됩니다.

```text
submission_code/
  data/                            # open.zip의 data
    train/
    eval/
  submission_kit/                  # open.zip의 submission_kit
  pretrained/                      # pretrained 모델
    cosmos-predict2.5/             # 공식 소스, 고정 commit
    checkpoints/                   # 공개 Cosmos 기반 checkpoint
      tokenizer.pth
      robot/action-cond/
        38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt
        cr1_empty_string_text_embeddings.pt
  weights/
    v014_stride4_200k_soup.pt      # 실제 제출 추론 weight
```

대회 데이터, 공식 Submission Kit, 공개 Cosmos 소스와 기반 checkpoint는 포함되지 않습니다. 
주최 측 `open.zip`의 `data/`와 `submission_kit/`은 내용을 변경하지 않고 위 빈 폴더에 복사합니다.

ZIP을 해제하고 `submission_code/`로 이동한 뒤, `open.zip`을 해제한 실제 경로를
넣어 다음처럼 복사합니다.

```bash
cp -a /path/to/open/data/. data/
cp -a /path/to/open/submission_kit/. submission_kit/
```

실제 업로드한 최종 CSV 원본(`submission_final.csv`)은 용량 문제로 공개 저장소에서 제외했습니다.
새 추론은 항상 영상에서 `outputs/final/submission_features.csv`를 다시 생성합니다. 
공식 feature extractor의 CUDA 연산 환경에 따라 새 CSV의 마지막 소수 자릿수는 실제 업로드 원본과 달라질 수 있습니다.

## 2. 실행 환경

검증 환경은 Ubuntu 22.04.5와 NVIDIA driver 575.51.02입니다. 
학습·영상 생성은 Python 3.10.12, PyTorch 2.7.0+cu128이고 공식 CSV Kit은 충돌을 피하기 위해 Python 3.12.8, PyTorch 2.7.1+cu128 환경에서 별도로 실행했습니다. 
아래 명령은 모두 ZIP을 해제한 `submission_code/` 안에서 실행합니다.

### 2.1 Cosmos 소스와 pretrained checkpoint 다운로드

아래 명령은 공식 소스와 실행 환경을 `pretrained/` 아래에 만들고,
학습과 추론에 필요한 공개 checkpoint 세 개를 정확한 상대경로로 내려받습니다.
Hugging Face의 NVIDIA 모델 페이지에서 라이선스에 먼저 동의해야 합니다.

```bash
# 현재 위치가 submission_code/인지 확인
test -f train.py && test -d pretrained

git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git \
  pretrained/cosmos-predict2.5
git -C pretrained/cosmos-predict2.5 checkout \
  a2c298b0a3df3778b973fe65e9e58877b292d8a7

# 공식 Cosmos 가상환경 생성 및 재현 버전 적용
cd pretrained/cosmos-predict2.5
uv sync --python 3.10 --extra cu128
uv pip install --python .venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r ../../requirements_train_infer.txt
cd ../..

# NVIDIA 모델 페이지에서 라이선스 동의 후 공개 기반 checkpoint 채우기
pretrained/cosmos-predict2.5/.venv/bin/hf auth login
pretrained/cosmos-predict2.5/.venv/bin/hf download \
  nvidia/Cosmos-Predict2.5-2B \
  tokenizer.pth \
  robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt \
  robot/action-cond/cr1_empty_string_text_embeddings.pt \
  --local-dir pretrained/checkpoints

# 세 파일이 정확한 상대경로에 들어왔는지 확인
test -f pretrained/checkpoints/tokenizer.pth
test -f pretrained/checkpoints/robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt
test -f pretrained/checkpoints/robot/action-cond/cr1_empty_string_text_embeddings.pt
test "$(git -C pretrained/cosmos-predict2.5 rev-parse HEAD)" = \
  "a2c298b0a3df3778b973fe65e9e58877b292d8a7"
```

checkpoint를 이미 준비했거나 다른 위치에 둘 경우에는 같은 디렉터리 구조를
유지하고 `train.py` 및 `inference.py`에 `--checkpoint-root`를 지정할 수 있습니다.

### 2.2 공식 Submission Kit 전용 환경 설치

```bash
python3.12 -m venv .scoring-venv
.scoring-venv/bin/pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements_scoring_kit.txt
```

`requirements_train_infer.txt`는 공식 Cosmos 환경 위에 적용하는 고정 버전이며 공식 환경 전체를 대신하지 않습니다. `inference.py`는 기본적으로 `.scoring-venv/bin/python`을 자동 선택하고, 없으면 현재 Python에서 공식 Kit
의존성을 검사합니다. 다른 환경을 쓸 때만 `--scoring-python`을 지정합니다.
공식 Kit이 사용하는 공개 모델들은 최초 실행 시 내려받습니다.

## 3. 학습 완료 weight로 제출 CSV 복원

첨부된 최종 weight를 사용하여 제출 결과를 복원합니다.

```bash
pretrained/cosmos-predict2.5/.venv/bin/python inference.py \
  --weight-mode packaged --gpu 0
```

결과 파일은 `outputs/final/submission_features.csv`입니다. 중간 산출물은 다음과
같습니다.

```text
outputs/final/
  raw_videos/                      # Cosmos 원 생성 영상 216개
  videos/                          # 고정 후처리 결과 216개
  videos/routing.csv               # sample별 후처리 분기 기록
  submission_features.csv          # 최종 제출 CSV
```

다른 경로 배치에서는 다음 인자를 사용할 수 있습니다.

```bash
pretrained/cosmos-predict2.5/.venv/bin/python inference.py \
  --weight-mode packaged \
  --data-root /path/to/data \
  --submission-kit /path/to/submission_kit \
  --cosmos-repo /path/to/cosmos-predict2.5 \
  --checkpoint-root /path/to/checkpoints \
  --output-root /path/to/output \
  --gpu 0
```

## 4. 처음부터 학습하는 명령어

`data/train`으로 인덱스 검증, latent 전처리, Stage 1, train-only latent IDM, Stage 2, checkpoint soup을 진행합니다.

```bash
pretrained/cosmos-predict2.5/.venv/bin/python train.py \
  --train-root data/train \
  --checkpoint-root pretrained/checkpoints \
  --work-root work \
  --gpu 0 \
  --stage all
```

최종 재학습 weight는 `work/v014_stride4_200k_soup_retrained.pt`입니다.
`work/`는 학습 명령이 자동 생성합니다. 다른 split 또는 episode 순서로 만든 cache가
섞여 있으면 tensor schema, 저장 action, 누락·extra 파일 검사에서 중단합니다.
검증을 통과한 latent cache와 Stage 1·2 checkpoint는 재사용되며, latent IDM 단계
도중 중단된 경우에는 해당 단계를 처음부터 다시 실행합니다.

학습 구성은 다음과 같습니다.

| 단계 | 데이터 | 주요 설정 | 종료 |
|---|---|---|---|
| Stage 1 | stride 8, 104,368 windows | LoRA r32, batch 8, LR 5e-5, motion power 1 | step 4,375 |
| Latent IDM | Stage 1 train latent | batch 64, LR 3e-4, EMA 0.999 | step 25,000 |
| Stage 2 | stride 4, 203,165 windows | Stage 1 재개, AUX weight/probability 0.3/0.3 | step 25,000 |
| Soup | Stage 2 checkpoint | 23,125/23,750/24,375/25,000 평균 | 최종 weight |

LoRA는 28개 DiT block의 self/cross-attention과 MLP projection에 적용하며,
6D action embedder도 함께 학습합니다. 학습 파라미터는 약 113.4M입니다.
설정은 `configs/training_config.json`과 각 run의 `run_config.json`에 기록되고,
resume 시 설정이 다르면 중단합니다. 새 로그는 `work/logs/`에 생성됩니다.
학습 seed는 torch/numpy `20260807`, sampler `1234`로 고정했습니다.

학습을 진행한 weight의 학습 보존 기록은 다음과 같습니다.

- Stage 1: step 4,375, `logs/stage1_v1_lora_train.log`
- latent IDM: step 25,000, 이 stdout 로그는 미보존
- Stage 2: 절대 step 25,000, `logs/stage2_stride4_train.log`
- soup: step 23,125·23,750·24,375·25,000 평균, 이 stdout 로그는 미보존

IDM과 soup은 로그가 없는 대신 실제 실행 step과 모든 주요 hyperparameter를 위 표와 `configs/training_config.json`에 명시했습니다. 
새로 전체 학습하면 IDM과 soup을 포함한 모든 stdout이 `work/logs/`에 기록됩니다.

## 5. 재학습 weight로 추론

학습 산출물을 평가할 때만 다음 경로를 사용합니다.

```bash
pretrained/cosmos-predict2.5/.venv/bin/python inference.py \
  --weight-mode retrained \
  --output-root outputs/retrained \
  --gpu 0
```

결과 CSV는 `outputs/retrained/submission_features.csv`에 생성됩니다. 첨부 weight의
산출물과 섞이지 않도록 별도 출력 경로를 사용합니다.

## 6. 고정 추론/후처리 설정

생성 설정은 `configs/inference_config.json`과 코드에 고정했습니다.

- 입력: 제공 첫 이미지와 `(16, 6)` action condition
- 출력: 16 frames, 320×512, 6 fps
- sampler: Flow UniPC, 30 steps, shift 5
- seed 7, auto-guidance 0.6, LoRA scale 1
- RELADA, reference adapter, warped noise 미사용

후처리는 제공 action의 shoulder-lift 축(index 1) 16-frame 평균이 45도(제출 결과를 비교하여 선정)보다 큰 sample에만 background anchor를 적용합니다. 
anchor는 생성 영상과 제공 첫 이미지의 차이로 motion mask를 만들고, mask 밖 배경을 첫 이미지로 복원합니다. 
그 외 sample은 생성 MP4를 그대로 복사합니다. 분기에는 제공 action만 사용하며 sample ID, eval 정답 영상, 다른 sample 또는 Submission Kit feature는 사용하지 않습니다. 
모든 영상 처리가 끝난 뒤 공식 Submission Kit을 CSV 변환에만 적용합니다.

## 7. 파일 summary

- `train.py`: 단일 GPU 전체 학습 진입점
- `preprocess.py`: 단일 GPU 전처리 진입점
- `inference.py`: 첨부/재학습 weight 선택부터 CSV까지 추론 진입점
- `code/inference/postprocess.py`: action 기반 고정 후처리
- `weights/v014_stride4_200k_soup.pt`: 실제 제출 추론 최종 weight
- `logs/`: 보존된 Stage 1·Stage 2 학습 로그
- `requirements_*.txt`: 학습·추론 및 공식 Kit의 검증 라이브러리 버전
- `LICENSES/`: 파생 weight 재배포에 필요한 NVIDIA 라이선스

## 8. 라이선스

Licensed by NVIDIA Corporation under the NVIDIA Open Model License  
Built on NVIDIA Cosmos

첨부 weight는 NVIDIA Cosmos-Predict2.5-2B 기반 파생 weight입니다. 재배포 조건은
`LICENSES/NVIDIA_OPEN_MODEL_LICENSE.txt`를 확인하십시오.
