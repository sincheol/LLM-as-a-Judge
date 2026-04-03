# ⚖️ LLM-as-a-Judge: Positional Bias Evaluation

<div align="center">
  <img src="https://img.shields.io/badge/Model-Gemini%203%20Flash-blue?style=flat-square&logo=google" alt="Gemini 3 Flash"/>
  <img src="https://img.shields.io/badge/Language-Python%203.11%2B-blue?style=flat-square&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Status-Experiment%20Complete-success?style=flat-square" alt="Status"/>
</div>

<br>

이 레포지토리는 LLM을 활용하여 두 개의 텍스트를 상호 비교 평가(A/B Test, LLM-as-a-Judge)할 때 발생하는 **위치 편향(Positional Bias)** 을 검증하기 위한 자동화된 실험 도구와 그 분석 결과를 담고 있음.

본 실험에서는 `gemini-3-flash-preview` 모델을 사용하여, **"내용의 품질과 무관하게 모델이 프롬프트 상의 특정 위치(첫 번째 vs 두 번째)에 배치된 선택지를 무조건 더 선호하는가?"** 에 대한 통계적 검증을 수행함.

---

## 핵심 요약 (TL;DR)

`gemini-3-flash-preview` 모델은 두 개의 옵션을 비교할 때, **두 번째 옵션을 강하게 선호(68% 승률)** 하는 강력한 위치 편향을 가지고 있음. 
편향 완화 전략(이중 평가, 랜덤 셔플링, 절대 점수 산정 등) 없이는 해당 모델을 **쌍별 비교(Pairwise Comparison) 로직에 단일 스텝으로 사용하는 것은 부적절함**.

[실험 결과](./RESULT.md)

---

## 실험 설계 및 방법론

### 1. 주제 및 비교 대상
- **질문**: "운영체제에서 프로세스란 무엇인가?"
- **Answer A**: 실행 단위 및 메모리 구조(Code, Data, Stack)에 초점을 맞춘 완성도 높은 답변
- **Answer B**: 생명주기(Lifecycle) 및 PCB 관리에 초점을 맞춘 답변

### 2. 실험 환경
- **총 시행 횟수**: 100회
- **그룹 분리**: 
  - 그룹 1 (`A-B` 순서로 제시) : 50회
  - 그룹 2 (`B-A` 순서로 제시) : 50회
- **모델**: `gemini-3-flash-preview` 
- **제한 사항**: API Rate Limiting 방지를 위한 시행 간 1초 지연(Sleep)

---

## 상세 실험 결과

전체 100회의 실험 로그 원본은 [`results.json`](./results.json) 파일에서 확인할 수 있음.

### 1. 위치별 승률 통계
| 지표 | 결과 |
|------|-----|
| 총 시행 횟수 | 100회 |
| API 파싱 또는 네트워크 오류 | 0건 |
| **첫 번째 위치** 승리 횟수 | 32회 (32%) |
| **두 번째 위치** 승리 횟수 | **68회 (68%)** ⚠️ |

*(이론적으로 위치 편향이 없다면 50%에 수렴해야 함. 중립 대비 **+18%p의 편향** 존재)*

### 2. 답변(Answer)별 교차 검증 성능
| 옵션 | 첫 번째 위치에 제시될 때 승률 | 두 번째 위치에 제시될 때 승률 |
|------|-------------------------------|-------------------------------|
| **Answer A** | 32/50 **(64%)** | 50/50 **(100%)** ⚠️ |
| **Answer B** | 0/50 **(0%)** ⚠️ | 18/50 **(36%)** |

**분석 포인트:**
답변 A가 품질이 더 좋기 때문에 전반적으로 승률이 높지만, **답변 B는 첫 번째 위치에 있을 때 단 한 번(0%)도 승리하지 못한 반면, 두 번째 위치에 배치되자 36% 확률로 승리**함. 이는 내용의 품질보다 텍스트의 배치 위치가 평가에 결정적인 영향을 미치고 있음을 확증함.

---

## 시사점 및 프로덕션 적용 가이드

이러한 실험 결과를 바탕으로, 프로젝트 및 프로덕션 환경에서 LLM을 채점관(Judge)으로 도입할 때 다음의 방어 로직 설계가 강력히 권장됨.

1. **이중 평가 (Dual-Evaluation)**
   - 프롬프트를 `[A vs B]` 로 한 번, `[B vs A]`로 또 한 번 호출하는 **비동기 투표 방식**을 채택함.
   - 두 판단이 일치할 때만 승리를 인정하고, 엇갈리면 무승부(Tie)로 처리함.
2. **위치 셔플링 (Position Shuffling)**
   - API 비용 문제로 단회성 호출만 불가능할 경우, A/B의 순서를 무작위(`random.shuffle`)로 배치하여 특정 참가자가 편향으로 인한 영구적인 손해/이득을 보지 못하도록 설계함.

---

## 빠른 시작 (Quick Start)

본 실험은 누구나 로컬에서 즉시 재현할 수 있음. 동일한 스크립트로 최신 모델들의 벤치마킹도 가능함.

```bash
# 1. 저장소 클론
git clone https://github.com/sincheol/LLM-as-a-Judge.git
cd LLM-as-a-Judge

# 2. 의존성 설치
pip install google-generativeai python-dotenv

# 3. 환경변수 셋팅 (.env 파일 생성)
echo "GEMINI_API_KEY=your-api-key-here" > .env

# 4. 자동화 테스트 스크립트 실행 (100번 호출)
python bias_test.py
```

## 파일 구조
- `bias_test.py`: 100회 자동화 A/B 테스트 스크립트 (Python)
- `results.json`: 100번의 LLM Raw Response 응답과 평가 사유가 담긴 원본 데이터

