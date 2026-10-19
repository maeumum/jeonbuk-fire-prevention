# 전북 화재 예방 우선지원 모델

**2026년 제4회 전북 청년 AI·빅데이터 경진대회 출품작**

전북특별자치도 14개 시군의 전기화재 데이터를 분석하여, 한정된 예방 자원을 어느 지역에 먼저 투입해야 하는지 데이터 기반으로 결정하는 우선지원 모델입니다.

## 실행 방법

### 1) 패키지 설치
```bash
pip install -r requirements.txt
```

### 2) API 키 설정 (선택)
```bash
cp .env.example .env
# .env 파일에 ANTHROPIC_API_KEY 입력
```

### 3) 앱 실행
```bash
streamlit run app.py
```

## 대시보드 구성

| 섹션 | 내용 |
|---|---|
| 전국 비교 | 시도별 화재율·추이·역설 분석·고령화 전망 |
| 연도별 추이 | 시군별 2021~2025 화재 건수 변화 |
| 위험지수 순위 | 종합위험점수 기반 14개 시군 순위 |
| 상관관계 검증 | 4단계 가설 검증 (r=0.152 → 0.854) |
| 회귀분석 | OLS 3단계 검증 (R²=0.730, p<0.001) |
| ML 위험 분류 | K-means 클러스터링 (k=3, 실루엣 0.45) |
| 지도로 보기 | Folium 기반 시군별 위험도 시각화 |
| 정책 시뮬레이터 | 자원배분 갭 분석 + 시뮬레이션 |
| AI 예방정책 브리핑 | Claude API 기반 시군별 맞춤 브리핑 |
| 예방 전략 | 핵심 발견 요약 + 정책 제언 |

## 기술 스택

- **Python** 3.10+, **Streamlit** 1.30+
- **pandas**, **plotly**, **statsmodels**, **scikit-learn**
- **folium** + **streamlit-folium**
- **Anthropic Claude API** (AI 브리핑)

## 데이터 출처

| 데이터 | 출처 |
|---|---|
| 전기화재 시군별 현황 | 한국전기안전공사 |
| 1인가구 현황 | 통계청 KOSIS |
| 노후주택 현황 | 국토교통부 |
| 비상소화장치 현황 | 전북특별자치도 소방본부 |
| 전국 소방서 좌표 | 소방청 |

## Streamlit Cloud 배포

1. 이 저장소를 fork
2. [share.streamlit.io](https://share.streamlit.io) 접속 → GitHub 연결
3. `app.py` 선택 → Deploy
4. Settings → Secrets에 `ANTHROPIC_API_KEY` 추가 (AI 브리핑 활성화)
