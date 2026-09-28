# 학습 완료 가중치

GitHub 100MB 제한으로 별도 배포합니다.

| 파일 | 크기 | 역할 |
|---|---:|---|
| `v014_stride4_200k_soup.pt` | 454MB | **최종 추론 가중치** (Stage 2 체크포인트 4개 Model Soup) |
| `stage1_v1_lora_35k.pt` | 433MB | Stage 1 결과 — Stage 2 재학습 시작점 |
| `lidm2_best.pt` | 16MB | Latent IDM 행동 판독기 — Stage 2 보조 손실용 |

다운로드: **(링크 추가 예정)**

받은 뒤 `weights/v014_stride4_200k_soup.pt` 위치에 두면 `inference.py --weight-mode packaged`가 바로 동작합니다.
