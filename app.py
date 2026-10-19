"""
전북 고령 1인가구 밀집지역 전기화재 예방 우선지원 모델
- 2026년 제4회 전북 청년 AI·빅데이터 경진대회 출품작
실행: streamlit run app.py
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import folium
from streamlit_folium import st_folium
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)


st.set_page_config(page_title="전북 화재 예방 우선지원 모델", page_icon="🚒", layout="wide")

# 이 스크립트 파일의 위치를 기준으로 data 폴더를 찾는다.
# (streamlit을 어느 디렉토리에서 실행하든 항상 정확한 경로를 찾기 위함)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ---------------- 데이터 로드 ----------------
@st.cache_data
def load_data():
    fire = pd.read_csv(DATA_DIR / "fire_raw.csv")
    summary = pd.read_csv(DATA_DIR / "summary.csv")
    return fire, summary

fire_df, summary_df = load_data()

@st.cache_data
def load_fire_devices():
    df = pd.read_csv(DATA_DIR / "비상소화장치현황.csv", encoding="cp949")
    df["시군"] = df["설치위치"].str.extract(r"전북특별자치도 (\S+시|\S+군)")
    return df

device_df = load_fire_devices()

@st.cache_data
def load_inspection():
    df = pd.read_csv(DATA_DIR / "정기점검_행정구역별.csv", encoding="cp949")
    # 2019년은 전국 기준 변경으로 이상치(20~30%대) — 비교 분석에서 제외
    return df[df["연도"] != 2019].copy()

inspection_df = load_inspection()

@st.cache_data
def load_fire_stations():
    import re
    df = pd.read_csv(DATA_DIR / "전국소방서_좌표.csv", encoding="cp949")
    jb = df[df["주소"].str.contains("전북|전라북도", na=False)].copy()
    def _sigun(addr):
        m = re.search(r"(전주시|군산시|익산시|정읍시|남원시|김제시|완주군|진안군|무주군|장수군|임실군|순창군|고창군|부안군)", addr)
        return m.group(1) if m else "기타"
    jb["시군"] = jb["주소"].apply(_sigun)
    return jb

station_df = load_fire_stations()

# 시군 중심 좌표 (시청/군청 기준 근사치)
CITY_COORDS = {
    "전주시": (35.8242, 127.1480), "군산시": (35.9676, 126.7369),
    "익산시": (35.9483, 126.9575), "정읍시": (35.5699, 126.8551),
    "남원시": (35.4164, 127.3905), "김제시": (35.8034, 126.8809),
    "완주군": (35.9046, 127.1622), "진안군": (35.7917, 127.4249),
    "무주군": (36.0069, 127.6608), "장수군": (35.6474, 127.5212),
    "임실군": (35.6178, 127.2892), "순창군": (35.3745, 127.1374),
    "고창군": (35.4351, 126.7020), "부안군": (35.7318, 126.7333),
}

@st.cache_resource
def fit_regression(df):
    """화재율에 대한 단순/다중회귀 모형을 적합한다."""
    y = df["천호당화재율"]

    # 중간 발견: 고령1인가구밀도
    X_simple = sm.add_constant(df[["고령1인가구밀도"]])
    m_simple = sm.OLS(y, X_simple).fit()

    # 구 다중회귀 (노후주택 허위상관 검증용)
    X_multi = sm.add_constant(df[["고령1인가구밀도", "노후주택비율"]])
    m_multi = sm.OLS(y, X_multi).fit()

    # 최종 발견: 고령 1인가구의 단독주택 거주율 (가장 강한 설명변수)
    X_new = sm.add_constant(df[["고령단독주택거주율"]])
    m_new = sm.OLS(y, X_new).fit()

    return m_simple, m_multi, m_new

reg_simple, reg_multi, reg_new = fit_regression(summary_df)

# 상관계수 동적 계산 (하드코딩 제거)
r_old_housing = round(float(summary_df["노후주택비율"].corr(summary_df["천호당화재율"])), 3)
r_elderly     = round(float(summary_df["고령1인가구밀도"].corr(summary_df["천호당화재율"])), 3)
r_single      = round(float(summary_df["고령단독주택거주율"].corr(summary_df["천호당화재율"])), 3)

@st.cache_data
def load_cause_data():
    return pd.read_csv(DATA_DIR / "fire_cause_by_year.csv", encoding="cp949")

cause_df = load_cause_data()

@st.cache_data
def load_old_housing():
    df = pd.read_csv(DATA_DIR / "노후주택현황.csv", encoding="cp949")
    df["시군"] = df["시군구"].str.replace("전주시완산구", "전주시").str.replace("전주시덕진구", "전주시")
    return df.groupby("시군")[["단독주택", "합계"]].sum().reset_index()

old_housing_df = load_old_housing()

# 노후 단독주택 비율 (30년 이상 단독주택 수 / 총주택수) — 아파트 제외 순수 단독 지표
_oh = summary_df[["시군", "총주택수"]].merge(old_housing_df, on="시군", how="left")
summary_df["노후단독주택비율"] = (_oh["단독주택"] / _oh["총주택수"] * 100).round(1).values
r_old_single = round(float(summary_df["노후단독주택비율"].corr(summary_df["천호당화재율"])), 3)

# ---------------- K-means 클러스터링 ----------------
CLUSTER_COLOR_MAP = {"고위험": "#d62728", "중위험": "#ff9800", "관리": "#4caf50"}

@st.cache_data
def run_kmeans(df):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    features = ["고령단독주택거주율", "천호당화재율", "고령1인가구밀도"]
    X = df[features].values
    X_scaled = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = round(float(silhouette_score(X_scaled, labels)), 3)
    # 천호당화재율 평균 내림차순으로 레이블 고정 → 재실행해도 "고위험" 그룹 일관성 유지
    rank_map = sorted(range(3), key=lambda c: X[labels == c, 1].mean(), reverse=True)
    name_map = {rank_map[0]: "고위험", rank_map[1]: "중위험", rank_map[2]: "관리"}
    result = df[["시군"]].copy()
    result["cluster_label"] = [name_map[l] for l in labels]
    return result, sil

cluster_df, sil_score = run_kmeans(summary_df)
summary_df = summary_df.merge(cluster_df, on="시군", how="left")

# ---------------- Folium 지도 렌더 함수 ----------------
def render_folium_map(summary_df, device_df, station_df, city_coords,
                      show_devices=True, show_stations=True, show_centers=False):
    m = folium.Map(location=[35.72, 127.15], zoom_start=9, tiles="OpenStreetMap")

    min_score = float(summary_df["종합위험점수"].min())
    max_score = float(summary_df["종합위험점수"].max())

    # 시군 위험지수 원
    for _, row in summary_df.iterrows():
        region = row["시군"]
        if region not in city_coords:
            continue
        lat, lng = city_coords[region]
        priority = int(row["우선순위"])
        color = "#d62728" if priority <= 3 else ("#ff9800" if priority <= 7 else "#4caf50")
        norm = 1 - (float(row["종합위험점수"]) - min_score) / (max_score - min_score + 1e-9)
        radius_m = int(2500 + norm * 5500)

        popup_html = (
            f"<b>{region}</b> — {priority}위<br>"
            f"천호당 화재율: <b>{row['천호당화재율']}건</b><br>"
            f"고령1인가구밀도: {int(row['고령1인가구밀도'])}<br>"
            f"종합위험점수: {int(row['종합위험점수'])}"
        )
        folium.Circle(
            location=[lat, lng],
            radius=radius_m,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.38,
            weight=2,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{region} ({priority}위)",
        ).add_to(m)
        # 시군 이름 레이블
        folium.Marker(
            [lat, lng],
            icon=folium.DivIcon(
                html=(
                    f'<div style="font-size:11px;font-weight:700;color:#222;'
                    f'white-space:nowrap;pointer-events:none;'
                    f'text-shadow:1px 1px 0 #fff,-1px 1px 0 #fff,'
                    f'1px -1px 0 #fff,-1px -1px 0 #fff;">{region}</div>'
                ),
                icon_size=(70, 20),
                icon_anchor=(35, 10),
            ),
        ).add_to(m)

    # 비상소화장치
    if show_devices:
        device_group = folium.FeatureGroup(name="비상소화장치", show=True)
        for _, row in device_df.iterrows():
            folium.CircleMarker(
                location=[float(row["위도"]), float(row["경도"])],
                radius=3,
                color="#1a6fb5",
                fill=True,
                fill_color="#1a6fb5",
                fill_opacity=0.85,
                weight=1,
                tooltip=str(row["설치위치"]),
            ).add_to(device_group)
        device_group.add_to(m)

    # 소방서
    if show_stations:
        station_group = folium.FeatureGroup(name="소방서", show=True)
        for _, row in station_df[station_df["유형"] == "소방서"].iterrows():
            folium.CircleMarker(
                location=[float(row["X좌표"]), float(row["Y좌표"])],
                radius=6,
                color="#b22222",
                fill=True,
                fill_color="#b22222",
                fill_opacity=0.9,
                weight=2,
                tooltip=str(row["소방서 및 안전센터명"]),
            ).add_to(station_group)
        station_group.add_to(m)

    # 119 안전센터
    if show_centers:
        center_group = folium.FeatureGroup(name="119 안전센터", show=True)
        for _, row in station_df[station_df["유형"] == "119안전센터"].iterrows():
            folium.CircleMarker(
                location=[float(row["X좌표"]), float(row["Y좌표"])],
                radius=4,
                color="#ff7f0e",
                fill=True,
                fill_color="#ff7f0e",
                fill_opacity=0.85,
                weight=1,
                tooltip=str(row["소방서 및 안전센터명"]),
            ).add_to(center_group)
        center_group.add_to(m)

    return m


# ---------------- AI 브리핑 생성 함수 ----------------
def generate_briefing(region, row):
    """
    생성형 AI API(Anthropic/OpenAI 등)를 연동하려면 아래 call_llm() 함수를 구현하세요.
    API 키가 없는 경우 규칙 기반 템플릿으로 대체됩니다.
    """
    try:
        return call_llm(region, row)
    except Exception:
        return template_briefing(region, row)


def call_llm(region, row, api_key=None):
    """Anthropic claude-sonnet-4-6 스트리밍 연동. generator를 반환합니다."""
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("API 키 없음")

    import anthropic

    _dev_cnt = int(device_df[device_df["시군"] == region].shape[0])
    _dev_density = round(_dev_cnt / float(row["총주택수"]) * 1000, 3) if float(row["총주택수"]) > 0 else 0
    _cluster = row.get("cluster_label", "정보 없음")
    _jb_avg_fire = round(float(summary_df["천호당화재율"].mean()), 3)

    system_prompt = """당신은 전북특별자치도 소방안전 정책 전문가입니다.
데이터 기반 분석 결과를 행정 현장에서 즉시 활용할 수 있는 형태로 정리합니다.
반드시 아래 형식으로만 출력하세요. 총 600자 이내로 간결하게 작성하세요.

### 진단 요약
(2~3문장: 해당 시군의 핵심 위험 구조와 ML 등급 의미 설명)

### 우선 개입 3가지
1. (구체적 정책 행동 — 담당 부서·사업명 포함)
2. (구체적 정책 행동)
3. (구체적 정책 행동)

### 근거
(데이터 수치 1~2개를 인용하여 위 제안의 통계적 근거 서술)"""

    user_prompt = f"""전북 {region} 전기화재 예방 정책 브리핑을 작성해주세요.

[위험 진단 데이터]
- 천호당 화재율: {row['천호당화재율']}건 (전북 평균 {_jb_avg_fire}건)
- 평균 재산피해: {row['평균피해액_천원']:,.0f}천원
- 고령(65세+) 1인가구 밀도: {row['고령1인가구밀도']}명/천호
- 고령 1인가구 단독주택 거주율: {row['고령단독주택거주율']}%
- 노후주택 비율: {row['노후주택비율']}%
- ML 위험 등급 (K-means, k=3, n=14시군): {_cluster}
- 비상소화장치: {_dev_cnt}개 (밀도 {_dev_density}개/천호)
- 전북 내 종합 우선순위: {int(row['우선순위'])}위 / 14개 시군

[분석 맥락]
회귀분석 결과 고령단독주택거주율이 천호당 화재율과 r=0.797로 가장 강한 설명변수입니다.
K-means 클러스터링(실루엣 점수 {sil_score:.3f})으로 {_cluster} 등급에 분류되었습니다."""

    client = anthropic.Anthropic(api_key=api_key)

    def _stream():
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=700,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text

    return _stream()


def template_briefing(region, row):
    """API 키가 없을 때 사용하는 규칙 기반 대체 브리핑"""
    priority = int(row["종합위험점수"])
    return f"""
### {region} 전기화재 예방정책 브리핑 (템플릿 생성)

**진단 요약**
- 천호당 화재율 {row['천호당화재율']:.2f}건, 고령 1인가구 밀도 {row['고령1인가구밀도']:.0f}(천호당)로
  전북 14개 시군 중 종합위험점수 {priority}점을 기록했습니다.

**제안 정책**
1. 고령 1인가구 밀집 지역(경로당·복지관 인근)을 중심으로 **무료 전기설비 안전점검 우선 배정**
2. 한국전기안전공사 지역사업소와 연계한 **찾아가는 전기안전교육** 운영
3. 노후 배선·누전차단기 교체비 지원사업 신청 안내 캠페인

> ⚠️ 이 브리핑은 API 키가 설정되지 않아 템플릿으로 생성되었습니다.
> 실제 생성형 AI 연동은 `ANTHROPIC_API_KEY` 환경변수를 설정하면 자동 활성화됩니다.
"""


# ---------------- 사이드바 ----------------
with st.sidebar:
    st.markdown("## 전북 화재 예방")
    st.caption("2026 전북 청년 AI·빅데이터 경진대회")
    st.divider()
    st.markdown("""
<div style="display:flex;flex-direction:column;gap:2px;">
  <p style="font-size:0.7rem;font-weight:700;color:#888;letter-spacing:0.08em;margin:12px 0 4px 0;">분석 보고서</p>
  <a href="#national"    style="display:block;padding:6px 10px;border-radius:6px;color:#333;text-decoration:none;font-size:0.9rem;">전국 비교</a>
  <a href="#trend"       style="display:block;padding:6px 10px;border-radius:6px;color:#333;text-decoration:none;font-size:0.9rem;">연도별 추이</a>
  <a href="#risk"        style="display:block;padding:6px 10px;border-radius:6px;color:#333;text-decoration:none;font-size:0.9rem;">위험지수 순위</a>
  <a href="#correlation" style="display:block;padding:6px 10px;border-radius:6px;color:#333;text-decoration:none;font-size:0.9rem;">상관관계 검증</a>
  <a href="#regression"  style="display:block;padding:6px 10px;border-radius:6px;color:#333;text-decoration:none;font-size:0.9rem;">회귀분석</a>
  <a href="#ml"          style="display:block;padding:6px 10px;border-radius:6px;color:#333;text-decoration:none;font-size:0.9rem;">ML 위험 분류</a>
  <p style="font-size:0.7rem;font-weight:700;color:#888;letter-spacing:0.08em;margin:16px 0 4px 0;">도구</p>
  <a href="#map"         style="display:block;padding:6px 10px;border-radius:6px;color:#333;text-decoration:none;font-size:0.9rem;">지도로 보기</a>
  <a href="#simulator"   style="display:block;padding:6px 10px;border-radius:6px;color:#333;text-decoration:none;font-size:0.9rem;">정책 시뮬레이터</a>
  <a href="#briefing"    style="display:block;padding:6px 10px;border-radius:6px;color:#333;text-decoration:none;font-size:0.9rem;">AI 예방정책 브리핑</a>
  <p style="font-size:0.7rem;font-weight:700;color:#888;letter-spacing:0.08em;margin:16px 0 4px 0;">결론</p>
  <a href="#conclusion"  style="display:block;padding:6px 10px;border-radius:6px;color:#333;text-decoration:none;font-size:0.9rem;">예방 전략</a>
</div>
""", unsafe_allow_html=True)

EFFECTIVE_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ---------------- 헤더 ----------------
st.markdown("""
<style>
/* 섹션 앵커 여백 */
.anchor-section { scroll-margin-top: 60px; }

/* 섹션 구분선 */
.section-sep {
    border: none;
    border-top: 2px solid #f0f0f0;
    margin: 3rem 0 2rem 0;
}

/* 섹션 레이블 */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: #888;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}

/* KPI 강조 숫자 */
.kpi-big {
    font-size: 2.8rem;
    font-weight: 800;
    color: #d62728;
    line-height: 1;
}
.kpi-label {
    font-size: 0.85rem;
    color: #666;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)
st.title("전북 화재 예방 우선지원 모델")
st.caption("2026년 제4회 전북 청년 AI·빅데이터 경진대회 | 데이터 출처: 한국전기안전공사, KOSIS 국가데이터처")

components.html("""
<script>
(function() {
  var SECTION_IDS = ['national','trend','risk','correlation','regression','ml','map','simulator','briefing','conclusion'];
  var ACTIVE_COLOR = '#d62728';
  var DEFAULT_COLOR = '#333';

  function getDoc() { return window.parent.document; }

  function setActive(activeId) {
    SECTION_IDS.forEach(function(id) {
      var link = getDoc().querySelector('a[href="#' + id + '"]');
      if (!link) return;
      link.style.color = (id === activeId) ? ACTIVE_COLOR : DEFAULT_COLOR;
      link.style.fontWeight = (id === activeId) ? '600' : 'normal';
    });
  }

  function init() {
    var doc = getDoc();
    var sections = SECTION_IDS.map(function(id) {
      return doc.getElementById(id);
    }).filter(Boolean);

    if (sections.length === 0) { setTimeout(init, 600); return; }

    if (window.parent._navObserver) window.parent._navObserver.disconnect();

    var observer = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) setActive(entry.target.id);
      });
    }, { root: null, rootMargin: '-10% 0px -65% 0px', threshold: 0 });

    window.parent._navObserver = observer;
    sections.forEach(function(s) { observer.observe(s); });
    setActive('national');
  }

  setTimeout(init, 1200);
})();
</script>
""", height=0, scrolling=False)

# ---------------- 상단 지표 카드 ----------------
col1, col2, col3, col4 = st.columns(4)
top1 = summary_df.sort_values("우선순위").iloc[0]
col1.metric("최우선 지역", top1["시군"], f"위험점수 {top1['종합위험점수']:.0f}")
col2.metric("전북 평균 천호당 화재율", f"{summary_df['천호당화재율'].mean():.2f}")
col3.metric("노후주택비율 상관계수", f"{r_old_housing:.2f}", "약함 → 아파트 포함 희석")
col4.metric("고령단독주택거주율 상관계수", f"{r_single:.2f}", "가장 강함 ★", delta_color="inverse")

st.divider()

# sorted_df는 여러 섹션에서 공통으로 사용하므로 앞에 정의
sorted_df = summary_df.sort_values("우선순위")

# ---------------- 섹션 9→1: 전국 비교 ----------------
st.markdown('<div class="anchor-section" id="national"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">분석 보고서</p>', unsafe_allow_html=True)
st.subheader("전북은 전국에서 얼마나 위험한가?")

@st.cache_data
def load_sido():
    raw = pd.read_csv(DATA_DIR / "sido_fire.csv", encoding="cp949", header=None)
    # 0행: 컬럼명 행, 1행~: 데이터
    # 연도별 건수 컬럼 위치: 2022=col1, 2023=col9, 2024=col17
    rows = raw.iloc[1:].copy()
    rows.columns = range(len(rows.columns))
    rows = rows.rename(columns={0: "시도"})
    rows["2022건수"] = pd.to_numeric(rows[1], errors="coerce")
    rows["2023건수"] = pd.to_numeric(rows[9], errors="coerce")
    rows["2024건수"] = pd.to_numeric(rows[17], errors="coerce")
    rows["평균건수"] = rows[["2022건수", "2023건수", "2024건수"]].mean(axis=1)
    rows = rows[["시도", "2022건수", "2023건수", "2024건수", "평균건수"]].dropna()
    rows = rows[rows["시도"] != "전국"].reset_index(drop=True)
    return rows

# 2024 시도별 추계인구 (통계청, 단위: 천명)
SIDO_POP = {
    "서울특별시": 9388, "부산광역시": 3298, "대구광역시": 2355, "인천광역시": 3047,
    "광주광역시": 1413, "대전광역시": 1437, "울산광역시": 1099, "세종특별자치시": 395,
    "경기도": 13820, "강원특별자치도": 1553, "충청북도": 1616, "충청남도": 2161,
    "전북특별자치도": 1753, "전라남도": 1800, "경상북도": 2574, "경상남도": 3247,
    "제주특별자치도": 676,
}

sido_df = load_sido()
sido_df["인구_천명"] = sido_df["시도"].map(SIDO_POP)
sido_df["10만명당화재율_2024"] = (sido_df["2024건수"] / sido_df["인구_천명"] * 100).round(1)
sido_df = sido_df.sort_values("10만명당화재율_2024", ascending=False).reset_index(drop=True)
sido_df["순위"] = sido_df.index + 1

jeonbuk_rank = int(sido_df[sido_df["시도"] == "전북특별자치도"]["순위"].values[0])
jeonbuk_rate = float(sido_df[sido_df["시도"] == "전북특별자치도"]["10만명당화재율_2024"].values[0])
national_rate = round(
    sido_df["2024건수"].sum() / sum(SIDO_POP.values()) * 100, 1
)

c1, c2, c3 = st.columns(3)
c1.metric("전북 전국 순위 (2024)", f"{jeonbuk_rank}위 / 17개 시도",
          "높을수록 화재 위험")
c2.metric("전북 10만명당 화재율", f"{jeonbuk_rate}건",
          f"전국 평균 대비 +{round(jeonbuk_rate - national_rate, 1)}건")
c3.metric("전국 평균 10만명당 화재율", f"{national_rate}건")

st.divider()

# 막대 그래프 — 시도별 10만명당 화재율
colors = ["#d62728" if r == "전북특별자치도" else "#aec7e8" for r in sido_df["시도"]]
fig_sido = go.Figure(go.Bar(
    x=sido_df["시도"],
    y=sido_df["10만명당화재율_2024"],
    marker_color=colors,
    text=sido_df["10만명당화재율_2024"],
    texttemplate="%{text}건",
    textposition="outside",
))
fig_sido.add_hline(
    y=national_rate, line_dash="dash", line_color="orange",
    annotation_text=f"전국 평균 {national_rate}건", annotation_position="top left"
)
fig_sido.update_layout(
    title="시도별 인구 10만명당 전기화재 건수 (2024)",
    xaxis_title="", yaxis_title="10만명당 화재건수",
    xaxis_tickangle=-30,
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_sido, use_container_width=True)

# 전북 vs 전국 평균 연도별 추이
st.divider()
st.markdown("#### 전북 vs 전국 평균 화재건수 추이 (2022~2024)")

national_avg = sido_df[["2022건수", "2023건수", "2024건수"]].sum() / 17
jb_row = sido_df[sido_df["시도"] == "전북특별자치도"].iloc[0]

trend_df = pd.DataFrame({
    "연도": [2022, 2023, 2024],
    "전북": [jb_row["2022건수"], jb_row["2023건수"], jb_row["2024건수"]],
    "전국 평균 (17개 시도)": [national_avg["2022건수"], national_avg["2023건수"], national_avg["2024건수"]],
})

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=trend_df["연도"], y=trend_df["전북"],
    mode="lines+markers+text", name="전북",
    line=dict(color="#d62728", width=3),
    text=trend_df["전북"].astype(int), textposition="top center",
))
fig_trend.add_trace(go.Scatter(
    x=trend_df["연도"], y=trend_df["전국 평균 (17개 시도)"],
    mode="lines+markers+text", name="전국 평균",
    line=dict(color="#aec7e8", width=2, dash="dash"),
    text=trend_df["전국 평균 (17개 시도)"].round(0).astype(int), textposition="top center",
))
fig_trend.update_layout(
    xaxis=dict(tickvals=[2022, 2023, 2024]),
    yaxis_title="화재건수 (건)",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_trend, use_container_width=True)

st.caption(
    f"전북 전국 {jeonbuk_rank}위 — 전국 평균 대비 {round((jeonbuk_rate/national_rate - 1)*100, 1)}% 높음. "
    "고령 1인가구 단독주택 밀집이라는 구조적 요인이 주원인."
)

st.divider()

@st.cache_data
def load_kesco_fire():
    df = pd.read_csv(DATA_DIR / "전기화재_지역별.csv", encoding="cp949")
    jb_regions = ["고창군", "군산시", "김제시", "남원시", "무주군", "부안군",
                  "순창군", "완주군", "익산시", "임실군", "장수군", "전주시",
                  "정읍시", "진안군"]
    # 2021년 전주시 완산구+덕진구 합산
    yr2021_jj = df[(df["연도"] == 2021) & (df["지역별"].str.contains("전주시"))]
    jj_sum = yr2021_jj[["전기화재 건수", "재산피해(천원)"]].sum()
    jj_row = pd.DataFrame({
        "지역별": ["전주시"], "연도": [2021],
        "전기화재 건수": [jj_sum["전기화재 건수"]],
        "재산피해(천원)": [jj_sum["재산피해(천원)"]],
    })
    rest = df[df["지역별"].isin(jb_regions)]
    return pd.concat([rest, jj_row], ignore_index=True).sort_values(["지역별", "연도"])

kesco_df = load_kesco_fire()
_years = sorted(kesco_df["연도"].unique())
st.markdown(f"#### 전북 14개 시군 연도별 전기화재 건수 추이 ({_years[0]}~{_years[-1]})")
st.caption("한국전기안전공사 연도별 지역 전기화재 현황")
_latest_yr = max(_years)
_prev_yr = _latest_yr - 1
total_by_yr = kesco_df.groupby("연도")["전기화재 건수"].sum()
dmg_by_yr = kesco_df.groupby("연도")["재산피해(천원)"].sum()

km1, km2, km3 = st.columns(3)
km1.metric(f"전북 전기화재 총건수 ({_latest_yr})",
           f"{int(total_by_yr[_latest_yr])}건",
           f"{int(total_by_yr[_latest_yr]) - int(total_by_yr[_prev_yr]):+d}건 (전년 대비)")
km2.metric(f"재산피해 ({_latest_yr})",
           f"{dmg_by_yr[_latest_yr]/1e6:.1f}억원")
km3.metric("5년 평균", f"{int(total_by_yr.mean())}건 / 년")

# 전북 전체 연도별 추이 바 차트
fig_total_fire = go.Figure(go.Bar(
    x=list(total_by_yr.index),
    y=list(total_by_yr.values),
    marker_color=["#d62728" if y == _latest_yr else "#aec7e8" for y in total_by_yr.index],
    text=list(total_by_yr.values),
    textposition="outside",
))
fig_total_fire.update_layout(
    title="전북 전기화재 건수 연도별 추이",
    xaxis=dict(tickvals=_years),
    yaxis_title="전기화재 건수",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_total_fire, use_container_width=True)

# 시군별 히트맵
_pivot = kesco_df.pivot_table(
    index="지역별", columns="연도", values="전기화재 건수", aggfunc="sum"
).fillna(0)
_pivot_sorted = _pivot.sort_values(_latest_yr, ascending=True)

fig_heatmap = go.Figure(go.Heatmap(
    z=_pivot_sorted[_years].values,
    x=[str(y) for y in _years],
    y=_pivot_sorted.index.tolist(),
    colorscale="YlOrRd",
    text=_pivot_sorted[_years].values.astype(int),
    texttemplate="%{text}건",
    hoverongaps=False,
))
fig_heatmap.update_layout(
    title="시군별 연도별 전기화재 건수 히트맵",
    height=420,
    margin=dict(l=80),
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# 증감률 분석
_first_yr = min(_years)
_pivot["증감률"] = ((_pivot[_latest_yr] - _pivot[_first_yr]) / _pivot[_first_yr] * 100).round(1)
_pivot["절대증감"] = (_pivot[_latest_yr] - _pivot[_first_yr]).astype(int)

# 비율 급증 (농촌형: 50%↑)
_pct_rising = _pivot[_pivot["증감률"] > 50].sort_values("증감률", ascending=False)
# 절대 건수 급증 상위 3개
_abs_rising = _pivot.sort_values("절대증감", ascending=False).head(3)

col_r1, col_r2 = st.columns(2)
with col_r1:
    st.markdown("**비율 급증 시군 (2021→2025, +50%↑)**")
    if not _pct_rising.empty:
        for r, row in _pct_rising.iterrows():
            st.markdown(
                f"- **{r}**: {int(row[_first_yr])}건 → {int(row[_latest_yr])}건 "
                f"(**{row['증감률']:+.0f}%**)"
            )
with col_r2:
    st.markdown("**절대 건수 증가 상위 시군 (2021→2025)**")
    for r, row in _abs_rising.iterrows():
        st.markdown(
            f"- **{r}**: {int(row[_first_yr])}건 → {int(row[_latest_yr])}건 "
            f"(**{int(row['절대증감']):+d}건**)"
        )

st.caption("농촌(비율 급증 = 고령화 신호) vs 도시(절대 건수 증가) — 두 유형 모두 예방 우선순위 높음.")

st.divider()
st.markdown("#### 역설: 정기점검 부적합률은 평균인데 왜 화재율은 높은가?")

jb_insp = inspection_df[inspection_df["행정구역별"] == "전북"]
nat_insp = inspection_df.groupby("연도")["부적합률"].mean().round(2).reset_index()
nat_insp.columns = ["연도", "전국평균"]

latest = inspection_df[inspection_df["연도"] == inspection_df["연도"].max()]
jb_latest = float(jb_insp[jb_insp["연도"] == jb_insp["연도"].max()]["부적합률"].values[0])
nat_latest = round(latest["부적합률"].mean(), 2)
jb_rank_insp = int(latest.sort_values("부적합률", ascending=False).reset_index(drop=True)
                   .query("행정구역별 == '전북'").index[0]) + 1

ci1, ci2, ci3 = st.columns(3)
ci1.metric("전북 부적합률 (최신)", f"{jb_latest}%",
           f"전국 {'상위' if jb_latest > nat_latest else '하위'} 수준")
ci2.metric("전국 평균 부적합률", f"{nat_latest}%")
ci3.metric(f"전북 전국 순위 ({inspection_df['연도'].max()}년)",
           f"{jb_rank_insp}위 / {latest['행정구역별'].nunique()}개 지역",
           "낮을수록 부적합 많음")

# 전북 vs 전국 평균 부적합률 추이
insp_trend = jb_insp[["연도", "부적합률"]].rename(columns={"부적합률": "전북"}).merge(nat_insp, on="연도")
fig_insp = go.Figure()
fig_insp.add_trace(go.Scatter(
    x=insp_trend["연도"], y=insp_trend["전북"],
    mode="lines+markers+text", name="전북",
    line=dict(color="#d62728", width=3),
    text=insp_trend["전북"], textposition="top center",
))
fig_insp.add_trace(go.Scatter(
    x=insp_trend["연도"], y=insp_trend["전국평균"],
    mode="lines+markers+text", name="전국 평균",
    line=dict(color="#aec7e8", width=2, dash="dash"),
    text=insp_trend["전국평균"].round(2), textposition="bottom center",
))
fig_insp.update_layout(
    title="전기안전 정기점검 부적합률 추이 (전북 vs 전국 평균)",
    xaxis=dict(tickvals=insp_trend["연도"].tolist()),
    yaxis_title="부적합률 (%)",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_insp, use_container_width=True)

# 최신연도 지역별 부적합률 순위
fig_insp_bar = px.bar(
    latest.sort_values("부적합률", ascending=False),
    x="행정구역별", y="부적합률",
    color="부적합률", color_continuous_scale="RdYlGn_r",
    text="부적합률",
    title=f"지역별 전기안전 정기점검 부적합률 ({inspection_df['연도'].max()}년)",
    labels={"행정구역별": "", "부적합률": "부적합률 (%)"},
)
fig_insp_bar.update_traces(
    texttemplate="%{text}%", textposition="outside",
    marker_line_color=["#d62728" if r == "전북" else "rgba(0,0,0,0)"
                       for r in latest.sort_values("부적합률", ascending=False)["행정구역별"]],
    marker_line_width=3,
)
fig_insp_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
st.plotly_chart(fig_insp_bar, use_container_width=True)

st.warning(
    f"⚠️ **역설**: 부적합률 전북 {jb_rank_insp}위({jb_latest}% ≈ 전국 평균 {nat_latest}%)인데 화재율은 전국 상위권. "
    "기존 점검이 고령 1인가구 단독주택에 닿지 않고 있다는 증거 → **맞춤형 찾아가는 점검** 필요."
)

# ---- 인구 고령화 추계 ----
st.divider()
st.markdown("#### 예고된 위기: 고령화가 전기화재 위험을 구조적으로 키운다")

@st.cache_data
def load_elderly_trend():
    df = pd.read_csv(DATA_DIR / "kosis_elderly_single_household.csv", encoding="cp949")
    jb = df[
        (df["행정구역별(시군구)"].str.contains("전북", na=False)) &
        (df["성별"] == "계") &
        (df["항목"] == "1인가구")
    ]
    elderly_ages = ["65~69세", "70~74세", "75~79세", "80~84세", "85세이상"]
    elderly = jb[jb["연령별"].isin(elderly_ages)].copy()
    total = jb[jb["연령별"] == "합계"].copy()
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    year_cols = ["2020 년", "2021 년", "2022 년", "2023 년", "2024 년", "2025 년"]
    for col in year_cols:
        elderly[col] = pd.to_numeric(elderly[col].astype(str).str.replace(",", ""), errors="coerce")
        total[col]   = pd.to_numeric(total[col].astype(str).str.replace(",", ""),   errors="coerce")
    elderly_counts = elderly[year_cols].sum().values.astype("int64")
    total_counts   = total[year_cols].values.flatten().astype("int64")
    ratio = (elderly_counts / total_counts * 100).round(1)
    return pd.DataFrame({
        "연도": years,
        "65세+_1인가구": elderly_counts,
        "총_1인가구": total_counts,
        "65세+_비율": ratio,
    })

@st.cache_data
def load_proj_2052():
    df = pd.read_csv(DATA_DIR / "population_projection_2052.csv", encoding="cp949")
    jb = df[df["시도별(1)"].str.contains("전북|전라북도", na=False)]
    elderly_pop = float(jb[jb["인구종류별(1)"] == "고령인구(천명): 65세+"]["2052"].values[0])
    elderly_pct = float(jb[jb["인구종류별(1)"] == "- 구성비(%): 65세+"]["2052"].values[0])
    return elderly_pop, elderly_pct

elderly_trend = load_elderly_trend()
proj_pop, proj_pct = load_proj_2052()

count_2025 = int(elderly_trend.loc[elderly_trend["연도"] == 2025, "65세+_1인가구"].values[0])
count_2020 = int(elderly_trend.loc[elderly_trend["연도"] == 2020, "65세+_1인가구"].values[0])
ratio_2025 = float(elderly_trend.loc[elderly_trend["연도"] == 2025, "65세+_비율"].values[0])
ratio_2020 = float(elderly_trend.loc[elderly_trend["연도"] == 2020, "65세+_비율"].values[0])

ce1, ce2, ce3 = st.columns(3)
ce1.metric("전북 65세+ 1인가구 (2025)", f"{count_2025:,}명",
           f"+{count_2025 - count_2020:,}명 (5년간 증가)")
ce2.metric("전체 1인가구 中 고령 비율 (2025)", f"{ratio_2025}%",
           f"{ratio_2020}% → {ratio_2025}% (2020 대비)")
ce3.metric("2052년 전북 고령인구 전망", f"{proj_pop:.0f}천명",
           f"전체 인구의 {proj_pct}%")

fig_eld = go.Figure()
fig_eld.add_trace(go.Scatter(
    x=elderly_trend["연도"], y=elderly_trend["65세+_1인가구"],
    mode="lines+markers+text", name="65세+ 1인가구",
    line=dict(color="#d62728", width=3),
    text=elderly_trend["65세+_1인가구"].apply(lambda x: f"{x:,}"),
    textposition="top center",
))
fig_eld.add_trace(go.Scatter(
    x=elderly_trend["연도"], y=elderly_trend["총_1인가구"],
    mode="lines+markers+text", name="전체 1인가구",
    line=dict(color="#aec7e8", width=2, dash="dash"),
    text=elderly_trend["총_1인가구"].apply(lambda x: f"{x:,}"),
    textposition="bottom center",
))
fig_eld.update_layout(
    title="전북 65세+ 1인가구 증가 추이 (2020~2025)",
    xaxis=dict(tickvals=elderly_trend["연도"].tolist()),
    yaxis_title="가구 수",
    plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_eld, use_container_width=True)

fig_ratio = px.bar(
    elderly_trend, x="연도", y="65세+_비율",
    color="65세+_비율", color_continuous_scale="Reds",
    text="65세+_비율",
    title="전북 전체 1인가구 중 65세+ 비율 추이 (%)",
    labels={"65세+_비율": "비율 (%)"},
)
fig_ratio.update_traces(texttemplate="%{text}%", textposition="outside")
fig_ratio.update_layout(
    plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False,
    xaxis=dict(tickvals=elderly_trend["연도"].tolist()),
)
st.plotly_chart(fig_ratio, use_container_width=True)

st.caption(f"2052년 전북 고령인구 전망 (통계청 중위): {proj_pop:.0f}천명 · 전체의 {proj_pct}%")

st.warning(
    f"고령단독주택거주율(r={r_single:.3f})이 화재율의 가장 강한 설명변수이므로, "
    f"고령인구 비율이 {proj_pct}%로 전망되는 2052년에는 현재보다 화재 위험이 높아질 가능성이 있습니다."
)

# ================================================================
# ---------------- 섹션 4: 연도별 추이 ----------------
st.markdown('<hr class="section-sep"><div class="anchor-section" id="trend"></div>', unsafe_allow_html=True)
st.subheader(f"시군별 전기화재 건수 추이 ({fire_df['연도'].min()}~{fire_df['연도'].max()})")
selected_regions = st.multiselect(
    "비교할 시군 선택", options=summary_df["시군"].tolist(),
    default=sorted_df["시군"].tolist()[:5]
)
if selected_regions:
    filtered = fire_df[fire_df["시군"].isin(selected_regions)]
    fig3 = px.line(filtered, x="연도", y="화재건수", color="시군", markers=True)
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("시군을 1개 이상 선택해주세요.")

# ---------------- 섹션 5: 위험지수 순위 ----------------
st.markdown('<hr class="section-sep"><div class="anchor-section" id="risk"></div>', unsafe_allow_html=True)
st.subheader("시군별 종합 위험지수 순위")
st.caption("화재율 · 피해액 · 고령밀도 순위 합산 → 낮을수록 시급")

fig = px.bar(
    sorted_df, x="시군", y="종합위험점수",
    color="종합위험점수", color_continuous_scale="RdYlGn",
    text="우선순위",
    labels={"종합위험점수": "종합위험점수 (낮을수록 시급)"}
)
fig.update_traces(texttemplate="%{text}순위", textposition="outside")
fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': sorted_df["시군"].tolist()})
st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    sorted_df[["우선순위", "시군", "천호당화재율", "평균피해액_천원", "고령1인가구밀도",
                "고령단독주택거주율", "노후주택비율", "종합위험점수"]]
    .rename(columns={"평균피해액_천원": "평균피해액(천원)", "고령단독주택거주율": "고령단독주택거주율(%)"}),
    use_container_width=True, hide_index=True
)

# ---------------- 섹션 6: 상관관계 ----------------
st.markdown('<hr class="section-sep"><div class="anchor-section" id="correlation"></div>', unsafe_allow_html=True)
st.subheader("상관관계 검증")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("**❌ 가설 1** — 노후주택 전체")
    st.metric("r", f"{r_old_housing:.2f}", "기각", delta_color="off")
    fig1 = px.scatter(
        summary_df, x="노후주택비율", y="천호당화재율", text="시군",
        trendline="ols", labels={"노후주택비율": "노후주택비율(%)", "천호당화재율": "천호당 화재율"}
    )
    fig1.update_traces(textposition="top center")
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.markdown("**가설 2** — 노후 단독주택")
    st.metric("r", f"{r_old_single:.2f}", "상승")
    fig_os = px.scatter(
        summary_df, x="노후단독주택비율", y="천호당화재율", text="시군",
        trendline="ols",
        labels={"노후단독주택비율": "노후단독주택비율(%)", "천호당화재율": "천호당 화재율"}
    )
    fig_os.update_traces(textposition="top center", marker_color="darkorange")
    st.plotly_chart(fig_os, use_container_width=True)

with c3:
    st.markdown("**가설 3** — 고령 1인가구 밀도")
    st.metric("r", f"{r_elderly:.2f}", "강함")
    fig2 = px.scatter(
        summary_df, x="고령1인가구밀도", y="천호당화재율", text="시군",
        trendline="ols", labels={"고령1인가구밀도": "고령1인가구밀도(천호당)", "천호당화재율": "천호당 화재율"}
    )
    fig2.update_traces(textposition="top center", marker_color="firebrick")
    st.plotly_chart(fig2, use_container_width=True)

with c4:
    st.markdown("**✅ 가설 4** — 고령 단독주택 거주율")
    st.metric("r", f"{r_single:.2f}", "최강")
    fig3 = px.scatter(
        summary_df, x="고령단독주택거주율", y="천호당화재율", text="시군",
        trendline="ols",
        labels={"고령단독주택거주율": "고령1인가구 단독주택 거주율(%)", "천호당화재율": "천호당 화재율"}
    )
    fig3.update_traces(textposition="top center", marker_color="darkred")
    _sc = summary_df[summary_df["시군"] == "순창군"]
    if not _sc.empty:
        fig3.add_annotation(
            x=float(_sc["고령단독주택거주율"].values[0]),
            y=float(_sc["천호당화재율"].values[0]),
            text="⚠️ 순창군<br>총주택수 적어<br>회귀선 아래 위치",
            showarrow=True, arrowhead=2, ax=50, ay=-50,
            font=dict(color="darkorange", size=10),
            bgcolor="lightyellow", bordercolor="darkorange",
        )
    st.plotly_chart(fig3, use_container_width=True)

st.info(
    f"노후주택 전체(r={r_old_housing:.2f}) → 단독 분리(r={r_old_single:.2f}) → "
    f"고령밀도(r={r_elderly:.2f}) → **고령 단독주택 거주율(r={r_single:.2f})** 순으로 설명력 강화. "
    "단독주택은 거주자가 직접 배선을 관리해야 하는 구조적 요인입니다."
)

with st.expander("순창군 이상치 해설", expanded=False):
    _sc_row = summary_df[summary_df["시군"] == "순창군"].iloc[0] if "순창군" in summary_df["시군"].values else None
    if _sc_row is not None:
        st.markdown(f"""
- 고령단독주택거주율 **{_sc_row['고령단독주택거주율']}%** · 천호당 화재율 **{_sc_row['천호당화재율']}건** · 총 주택수 **{int(_sc_row['총주택수']):,}호**
- 총 주택수가 적어 절대 건수가 소수 → 화재율이 회귀선 아래 위치. 구조적 위험이 수치에 아직 반영되지 않은 예비 위험지역.
""")
    else:
        st.caption("순창군 데이터 없음")


# ---------------- 섹션 7: 회귀분석 ----------------
st.markdown('<hr class="section-sep"><div class="anchor-section" id="regression"></div>', unsafe_allow_html=True)
st.subheader("회귀분석: 무엇이 화재율을 통계적으로 설명하는가")

with st.expander("배경: 전국 전기화재 발화원인 통계 — '관리·점검 소홀형' 원인의 증가 (2022~2025)", expanded=False):
    _배방치형 = ["절연열화에 의한 단락", "트래킹에 의한 단락"]
    _관리소홀 = ["절연열화에 의한 단락", "트래킹에 의한 단락", "과부하 및 과전류", "접촉 불량"]
    _years = sorted(cause_df["연도"].unique())
    cause_years = [str(y) for y in _years]
    aging_cause_share = [
        round(cause_df[(cause_df["연도"] == y) & (cause_df["발화 원인"].isin(_배방치형))]["점유율"].sum(), 1)
        for y in _years
    ]
    neglect_total = [
        round(cause_df[(cause_df["연도"] == y) & (cause_df["발화 원인"].isin(_관리소홀))]["점유율"].sum(), 1)
        for y in _years
    ]
    _last_neglect_cnt = int(cause_df[
        (cause_df["연도"] == _years[-1]) & (cause_df["발화 원인"].isin(_관리소홀))
    ]["화재 건수"].sum())

    cause_fig = go.Figure()
    cause_fig.add_trace(go.Scatter(
        x=cause_years, y=aging_cause_share, mode="lines+markers",
        name="배선 방치형 (절연열화+트래킹)", line=dict(color="firebrick")
    ))
    cause_fig.add_trace(go.Scatter(
        x=cause_years, y=neglect_total, mode="lines+markers",
        name="관리소홀 합산 (방치형+과부하+접촉불량)", line=dict(color="darkred", dash="dash")
    ))
    cause_fig.update_layout(
        title="전국 전기화재 중 '관리·점검 소홀형' 원인 비중 추이",
        yaxis_title="점유율(%)", xaxis_title="연도"
    )
    st.plotly_chart(cause_fig, use_container_width=True)
    st.caption(
        f"배선 방치형 {aging_cause_share[0]}% → {aging_cause_share[-1]}% / "
        f"관리소홀 합산 {neglect_total[0]}% → {neglect_total[-1]}% "
        f"({cause_years[-1]}년 {_last_neglect_cnt:,}건) — 고령 1인가구 단독주택 밀집지역 집중 예방의 근거."
    )

st.markdown("#### 세 단계 검증")

step1, step2, step3 = st.columns(3)
with step1:
    st.markdown("**STEP 1** — 화재율 ~ 노후주택비율")
    pv1 = reg_multi.pvalues["노후주택비율"]
    st.metric("상관계수", f"{r_old_housing:.3f}", delta="기각", delta_color="off")
    st.metric("회귀 p-value", f"{pv1:.4f}", "유의하지 않음", delta_color="off")
    st.caption("아파트 포함으로 지표 희석 → 기각")

with step2:
    st.markdown("**STEP 2** — 화재율 ~ 고령1인가구밀도")
    pv2 = reg_simple.pvalues["고령1인가구밀도"]
    st.metric("상관계수", f"{r_elderly:.3f}", delta="강한 양의 상관")
    st.metric("R²", f"{reg_simple.rsquared:.3f}")
    st.metric("p-value", f"{pv2:.4f}", "통계적 유의(p<0.01)")
    st.caption("'왜?'라는 메커니즘 질문이 남음")

with step3:
    st.markdown(f"**STEP 3** — 화재율 ~ 고령단독주택거주율 ⭐")
    pv3 = reg_new.pvalues["고령단독주택거주율"]
    st.metric("상관계수", f"{r_single:.3f}", delta="최강 설명력")
    st.metric("R²", f"{reg_new.rsquared:.3f}")
    st.metric("p-value", f"{pv3:.4f}", "통계적 유의(p<0.01)")
    st.caption("인과 메커니즘 완성")

st.divider()
st.markdown("#### 최종 회귀모형: 화재율 ~ 고령 1인가구 단독주택 거주율")

coef3 = reg_new.params["고령단독주택거주율"]
const3 = reg_new.params["const"]
ci3 = reg_new.conf_int().loc["고령단독주택거주율"]
st.code(f"천호당 화재율 = {const3:.4f} + {coef3:.5f} × 고령단독주택거주율(%)", language=None)
st.caption(
    f"거주율 1%p 상승 → 화재율 +{coef3:.4f}건 "
    f"(95% CI [{ci3[0]:.4f}, {ci3[1]:.4f}], R²={reg_new.rsquared:.3f}, p={pv3:.4f})"
)

st.divider()
st.markdown("#### 95% 신뢰구간 — 3단계 모형 비교")
st.caption("0을 포함하지 않으면 통계적으로 유의")

_ci_rows = []
for model, var, label in [
    (reg_simple, "고령1인가구밀도",   "STEP 2: 고령1인가구밀도"),
    (reg_new,    "고령단독주택거주율", "STEP 3: 고령단독주택거주율 ⭐"),
    (reg_multi,  "노후주택비율",       "STEP 1: 노후주택비율 (다중회귀 내)"),
]:
    ci = model.conf_int().loc[var]
    coef = model.params[var]
    pv   = model.pvalues[var]
    _ci_rows.append({"변수": label, "계수": round(coef, 5),
                     "95% CI 하한": round(ci[0], 5), "95% CI 상한": round(ci[1], 5),
                     "p-value": round(pv, 4),
                     "유의": "✅" if pv < 0.05 else "❌"})
ci_df = pd.DataFrame(_ci_rows)

fig_ci = go.Figure()
colors = ["#d62728", "#d62728", "#aaa"]
for i, row in ci_df.iterrows():
    fig_ci.add_trace(go.Scatter(
        x=[row["95% CI 하한"], row["계수"], row["95% CI 상한"]],
        y=[row["변수"]] * 3,
        mode="lines+markers",
        marker=dict(size=[6, 12, 6], color=colors[i]),
        line=dict(color=colors[i], width=2),
        name=row["변수"], showlegend=False,
    ))
fig_ci.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="0")
fig_ci.update_layout(
    xaxis_title="회귀계수 (95% 신뢰구간)", yaxis_title="",
    height=200, margin=dict(l=20, r=20, t=20, b=20),
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_ci, use_container_width=True)
st.dataframe(ci_df, use_container_width=True, hide_index=True)

st.divider()
with st.expander("참고: STEP 2의 다중회귀 — 노후주택 허위상관 검증", expanded=False):
    st.markdown("**화재율 ~ 고령1인가구밀도 + 노후주택비율** 동시 투입 시:")
    ci_multi = reg_multi.conf_int().round(4)
    coef_table = pd.DataFrame({
        "회귀계수": reg_multi.params,
        "95% CI 하한": ci_multi[0],
        "95% CI 상한": ci_multi[1],
        "p-value": reg_multi.pvalues,
    }).round(4)
    coef_table["유의성"] = coef_table["p-value"].apply(
        lambda p: "✅ 유의(p<0.05)" if p < 0.05 else "- 유의하지 않음"
    )
    st.dataframe(coef_table, use_container_width=True)
    st.caption("고령밀도만 유의(p=0.001) — 노후주택비율은 고령밀도에 의한 허위상관.")

st.divider()
_r_multicollin = round(float(summary_df["고령단독주택거주율"].corr(summary_df["고령1인가구밀도"])), 3)
st.caption(
    f"한계: 표본 14개 시군 · 두 변수 간 다중공선성(r={_r_multicollin}) → "
    "단독주택거주율을 최종 변수로 채택. 상관관계이며 인과의 직접 증명은 아닙니다."
)


# ---------------- 섹션 8: ML 분류 ----------------
st.markdown('<hr class="section-sep"><div class="anchor-section" id="ml"></div>', unsafe_allow_html=True)
st.subheader("ML 위험 등급 분류 — K-means 클러스터링")
st.caption("비지도학습(Unsupervised Learning)으로 전북 14개 시군을 구조적 유사성 기반 3개 등급으로 분류합니다.")

st.caption("n=14 소표본 → train/test 분리 불가, K-means 비지도학습 채택. 클러스터 경계는 패턴 참고용.")

_n_high = int((summary_df["cluster_label"] == "고위험").sum())
_n_mid  = int((summary_df["cluster_label"] == "중위험").sum())
_n_low  = int((summary_df["cluster_label"] == "관리").sum())

km1, km2, km3, km4 = st.columns(4)
km1.metric("실루엣 점수", f"{sil_score:.3f}", "0.4↑ = 유의미한 분리")
km2.metric("고위험 시군", f"{_n_high}개", "즉각 개입 필요")
km3.metric("중위험 시군", f"{_n_mid}개", "선제적 모니터링")
km4.metric("관리 시군", f"{_n_low}개", "현 체계 유지")

# 버블 산점도 — 3번째 특성(고령밀도)을 버블 크기로 표현
fig_km = px.scatter(
    summary_df,
    x="고령단독주택거주율", y="천호당화재율",
    color="cluster_label", text="시군",
    size="고령1인가구밀도",
    color_discrete_map=CLUSTER_COLOR_MAP,
    labels={
        "고령단독주택거주율": "고령 1인가구 단독주택 거주율 (%)",
        "천호당화재율": "천호당 전기화재율 (건)",
        "cluster_label": "ML 위험 등급",
        "고령1인가구밀도": "고령 1인가구 밀도",
    },
    title="K-means 클러스터링 결과 (k=3, 특성: 고령단독거주율·화재율·고령밀도)",
)
fig_km.update_traces(textposition="top center", marker_opacity=0.85)
fig_km.update_layout(plot_bgcolor="rgba(0,0,0,0)", legend_title="ML 위험 등급")
st.plotly_chart(fig_km, use_container_width=True)

# 시군별 클러스터 결과 테이블
_km_display = summary_df[[
    "cluster_label", "시군", "고령단독주택거주율",
    "천호당화재율", "고령1인가구밀도", "우선순위"
]].sort_values(["cluster_label", "천호당화재율"], ascending=[True, False])
_km_display = _km_display.rename(columns={
    "cluster_label": "ML 위험 등급",
    "고령단독주택거주율": "고령단독거주율(%)",
    "고령1인가구밀도": "고령밀도(천호당)",
})
st.dataframe(_km_display, use_container_width=True, hide_index=True)

st.caption("회귀(평균 경향) + 클러스터(구조적 유사 그룹) 결합 → 위험도 측정 + 정책 묶음 설계 동시 가능.")


# ---------------- 섹션 1→7: 지도 ----------------
st.markdown('<hr class="section-sep"><div class="anchor-section" id="map"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">도구</p>', unsafe_allow_html=True)
st.subheader("시군별 위험지수 지도")
st.caption("원의 크기 = 종합위험점수(클수록 시급), 색상 = 우선순위 등급 · 원 클릭 시 상세 정보")

lc1, lc2, lc3 = st.columns(3)
show_devices  = lc1.checkbox("비상소화장치", value=True)
show_stations = lc2.checkbox("소방서", value=True)
show_centers  = lc3.checkbox("119 안전센터", value=False)

folium_map = render_folium_map(
    summary_df, device_df, station_df, CITY_COORDS,
    show_devices=show_devices,
    show_stations=show_stations,
    show_centers=show_centers,
)
st_folium(folium_map, use_container_width=True, height=570)

st.caption("빨강 = 우선순위 1–3위 · 주황 = 4–7위 · 초록 = 8위 이하 · 파랑 = 비상소화장치 · 진빨강 = 소방서")

st.divider()
st.markdown("#### 화재 위험순위 vs 비상소화장치 보급 현황")
st.caption("좌측 하단(위험 높음 + 장치 적음)에 위치한 시군이 인프라 보강 우선 지역입니다.")

device_count = device_df.groupby("시군").size().reset_index(name="비상소화장치수")
station_count = station_df.groupby("시군").size().reset_index(name="소방관서수")
risk_device = (
    summary_df[["시군", "우선순위", "천호당화재율", "고령단독주택거주율"]]
    .merge(device_count, on="시군", how="left")
    .merge(station_count, on="시군", how="left")
)
risk_device["비상소화장치수"] = risk_device["비상소화장치수"].fillna(0).astype(int)
risk_device["소방관서수"] = risk_device["소방관서수"].fillna(0).astype(int)
risk_device = risk_device.sort_values("우선순위")

fig_device = px.scatter(
    risk_device, x="우선순위", y="소방관서수", text="시군",
    size="천호당화재율", color="고령단독주택거주율",
    color_continuous_scale="RdYlGn_r",
    labels={
        "우선순위": "화재위험 우선순위 (낮을수록 위험)",
        "소방관서수": "소방관서 수 (개, 소방서+119센터)",
        "고령단독주택거주율": "고령단독주택거주율(%)",
    }
)
fig_device.update_traces(textposition="top center")
fig_device.update_xaxes(autorange="reversed")
st.plotly_chart(fig_device, use_container_width=True)

st.dataframe(
    risk_device[["우선순위", "시군", "천호당화재율", "고령단독주택거주율", "소방관서수", "비상소화장치수"]].rename(
        columns={
            "고령단독주택거주율": "고령단독주택거주율(%)",
            "소방관서수": "소방관서수(개)",
            "비상소화장치수": "비상소화장치수(개)",
        }
    ),
    use_container_width=True, hide_index=True
)
st.caption("좌하단(위험 높음 + 소방관서 적음) = 이중 취약 — 진안·임실·무주·순창은 고위험이나 소방관서 1~2개.")

# ---------------- 섹션 3: 정책 시뮬레이터 ----------------
st.markdown('<hr class="section-sep"><div class="anchor-section" id="simulator"></div>', unsafe_allow_html=True)
st.subheader("예방점검 자원배분 정책 시뮬레이터")

sim_region = st.selectbox("점검 인력을 우선 투입할 시군 선택", options=summary_df["시군"].tolist(), key="sim_region")
intervention_pct = st.slider(
    "점검 강화로 인한 화재율 감소 효과(가정)", min_value=0, max_value=50, value=20, step=5,
    help="전기안전공사 점검 강화가 화재 발생을 몇 % 줄인다고 가정할지 설정합니다. "
         "실제 감소율은 사업 시행 후 재검증이 필요합니다."
) / 100

sim_df = summary_df.copy()
idx = sim_df["시군"] == sim_region
sim_df.loc[idx, "천호당화재율_시뮬레이션"] = sim_df.loc[idx, "천호당화재율"] * (1 - intervention_pct)
sim_df["천호당화재율_시뮬레이션"] = sim_df["천호당화재율_시뮬레이션"].fillna(sim_df["천호당화재율"])

sim_df["rank_화재율_후"] = sim_df["천호당화재율_시뮬레이션"].rank(ascending=False)
sim_df["종합위험점수_후"] = sim_df["rank_화재율_후"] + sim_df["rank_피해액"] + sim_df["rank_고령밀도"]
sim_df["우선순위_후"] = sim_df["종합위험점수_후"].rank(ascending=True).astype(int)

before_rank = int(sim_df.loc[idx, "우선순위"].values[0])
after_rank = int(sim_df.loc[idx, "우선순위_후"].values[0])

c1, c2, c3 = st.columns(3)
c1.metric("대상 지역", sim_region)
c2.metric("시뮬레이션 전 우선순위", f"{before_rank}위")
if after_rank > before_rank:
    _delta_str, _delta_color = f"{after_rank - before_rank}단계 개선", "normal"
elif after_rank < before_rank:
    _delta_str, _delta_color = f"{before_rank - after_rank}단계 악화", "inverse"
else:
    _delta_str, _delta_color = "변동 없음", "off"
c3.metric("시뮬레이션 후 우선순위", f"{after_rank}위", delta=_delta_str, delta_color=_delta_color)

compare_df = sim_df[["시군", "우선순위", "우선순위_후"]].sort_values("우선순위")
compare_df.columns = ["시군", "개입 전 순위", "개입 후 순위"]
st.dataframe(compare_df, use_container_width=True, hide_index=True)

st.caption("실제 감소 효과는 사업 시행 후 데이터로 재보정이 필요합니다.")

# ---- 자원 배분 갭 분석 ----
st.divider()
st.markdown("#### 비상소화장치 자원 배분 갭 분석")
st.caption("ML 위험 등급 × 장치밀도 교차 — 고위험+인프라 부족 시군 식별")

_dev_count = device_df.groupby("시군").size().reset_index(name="비상소화장치수")
gap_df = summary_df[[
    "시군", "총주택수", "우선순위", "천호당화재율", "cluster_label"
]].merge(_dev_count, on="시군", how="left")
gap_df["비상소화장치수"] = gap_df["비상소화장치수"].fillna(0).astype(int)
gap_df["장치밀도_천호당"] = (gap_df["비상소화장치수"] / gap_df["총주택수"] * 1000).round(3)
_dev_median = gap_df["장치밀도_천호당"].median()
gap_df["갭_위험"] = (
    (gap_df["cluster_label"] == "고위험") & (gap_df["장치밀도_천호당"] < _dev_median)
)

gm1, gm2, gm3 = st.columns(3)
gm1.metric("전북 평균 장치밀도 (천호당)", f"{gap_df['장치밀도_천호당'].mean():.3f}개")
gm2.metric("고위험 · 장치부족 시군", f"{int(gap_df['갭_위험'].sum())}개", "즉시 보강 필요")
gm3.metric("중앙값 미만 장치 시군", f"{int((gap_df['장치밀도_천호당'] < _dev_median).sum())}개")

fig_gap = px.scatter(
    gap_df,
    x="우선순위", y="장치밀도_천호당",
    color="cluster_label", text="시군",
    size="천호당화재율",
    color_discrete_map=CLUSTER_COLOR_MAP,
    labels={
        "우선순위": "화재위험 우선순위 (1위 = 가장 위험)",
        "장치밀도_천호당": "비상소화장치 밀도 (개/천호)",
        "cluster_label": "ML 위험 등급",
    },
    title="위험 우선순위 vs 비상소화장치 밀도 — 좌상단이 즉각 보강 대상",
)
fig_gap.update_xaxes(autorange="reversed")
fig_gap.add_hline(
    y=_dev_median, line_dash="dash", line_color="gray",
    annotation_text=f"중앙값 {_dev_median:.3f}개/천호",
    annotation_position="top right",
)
fig_gap.update_traces(textposition="top center", marker_opacity=0.85)
fig_gap.update_layout(plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_gap, use_container_width=True)

_gap_table = gap_df[[
    "시군", "cluster_label", "우선순위", "비상소화장치수",
    "장치밀도_천호당", "갭_위험"
]].sort_values("우선순위").rename(columns={
    "cluster_label": "ML 위험 등급",
    "장치밀도_천호당": "장치밀도(개/천호)",
    "갭_위험": "즉시보강필요",
})
st.dataframe(_gap_table, use_container_width=True, hide_index=True)

_gap_regions = gap_df[gap_df["갭_위험"]]["시군"].tolist()
if _gap_regions:
    st.error(f"🚨 **즉시 보강 필요**: **{', '.join(_gap_regions)}** — ML 고위험 + 장치밀도 중앙값 미만")

# ---------------- 섹션 2: AI 브리핑 ----------------
st.markdown('<hr class="section-sep"><div class="anchor-section" id="briefing"></div>', unsafe_allow_html=True)
st.subheader("시군별 AI 예방정책 브리핑")
st.caption("ML 위험 등급 + 갭 분석 포함 · Anthropic claude-sonnet-4-6")

region_choice = st.selectbox("시군 선택", options=sorted_df["시군"].tolist())
row = summary_df[summary_df["시군"] == region_choice].iloc[0]

# 선택 시군 KPI 카드
_cl = row.get("cluster_label", "N/A")
_cl_icon = {"고위험": "🔴", "중위험": "🟠", "관리": "🟢"}.get(_cl, "⚪")
_cl_color = CLUSTER_COLOR_MAP.get(_cl, "#888")

st.markdown(
    f"""
<div style="background:#f8f9fa;border-left:5px solid {_cl_color};
            border-radius:8px;padding:20px 24px;margin:12px 0 16px 0;">
  <div style="font-size:0.75rem;font-weight:700;color:#888;letter-spacing:0.1em;
              text-transform:uppercase;margin-bottom:12px;">선택 지역 진단</div>
  <div style="display:flex;gap:32px;flex-wrap:wrap;align-items:flex-end;">
    <div>
      <div style="font-size:2rem;font-weight:800;color:{_cl_color};line-height:1;">
        {_cl_icon} {_cl}
      </div>
      <div style="font-size:0.8rem;color:#666;margin-top:4px;">ML 위험 등급</div>
    </div>
    <div>
      <div style="font-size:2rem;font-weight:800;color:#222;line-height:1;">
        {int(row['우선순위'])}위
      </div>
      <div style="font-size:0.8rem;color:#666;margin-top:4px;">전북 내 우선순위 / 14</div>
    </div>
    <div>
      <div style="font-size:2rem;font-weight:800;color:#222;line-height:1;">
        {row['천호당화재율']}건
      </div>
      <div style="font-size:0.8rem;color:#666;margin-top:4px;">천호당 화재율</div>
    </div>
    <div>
      <div style="font-size:2rem;font-weight:800;color:#222;line-height:1;">
        {row['고령단독주택거주율']}%
      </div>
      <div style="font-size:0.8rem;color:#666;margin-top:4px;">고령단독주택거주율</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if st.button("AI 브리핑 생성", type="primary"):
    if EFFECTIVE_ANTHROPIC_KEY:
        try:
            stream_gen = call_llm(region_choice, row, api_key=EFFECTIVE_ANTHROPIC_KEY)
            st.write_stream(stream_gen)
        except Exception as e:
            st.warning(f"API 오류 ({e}). 템플릿 브리핑으로 대체합니다.")
            st.markdown(template_briefing(region_choice, row))
    else:
        st.warning("ANTHROPIC_API_KEY 환경변수가 없습니다. 템플릿 브리핑으로 대체합니다.")
        st.markdown(template_briefing(region_choice, row))

# 섹션 10: 결론 및 예방 전략
# ================================================================
st.markdown('<hr class="section-sep"><div class="anchor-section" id="conclusion"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">결론</p>', unsafe_allow_html=True)
st.subheader("분석 결과 종합 및 예방 전략")

# ── 핵심 발견 3가지 ──────────────────────────────────────────
_top3 = sorted_df.head(3)["시군"].tolist()
_high_risk = summary_df[summary_df["cluster_label"] == "고위험"]["시군"].tolist()
_gap_urgent = gap_df[gap_df["갭_위험"]]["시군"].tolist()

st.markdown("#### 핵심 발견")
f1, f2, f3 = st.columns(3)

with f1:
    st.markdown(
        f"""<div style="background:#fff5f5;border-radius:8px;padding:16px 18px;height:100%;">
<div style="font-size:1.5rem;margin-bottom:6px;">🔍</div>
<div style="font-weight:700;margin-bottom:6px;">단독주택 구조가 핵심 위험 요인</div>
<div style="font-size:0.88rem;color:#555;">
고령 1인가구 단독주택 거주율(r={r_single:.3f})이 천호당 화재율과 가장 강한 상관관계.
노후주택 전체(r={r_old_housing:.2f})보다 <b>단독주택 여부</b>가 더 결정적입니다.
</div></div>""", unsafe_allow_html=True)

with f2:
    st.markdown(
        f"""<div style="background:#fff8f0;border-radius:8px;padding:16px 18px;height:100%;">
<div style="font-size:1.5rem;margin-bottom:6px;">📊</div>
<div style="font-weight:700;margin-bottom:6px;">기존 점검 체계의 사각지대</div>
<div style="font-size:0.88rem;color:#555;">
전기안전 정기점검 부적합률은 전국 평균 수준임에도
화재율은 전국 {jeonbuk_rank}위. 현행 점검이 고령 1인가구 단독주택에
<b>닿지 않고 있다</b>는 구조적 증거입니다.
</div></div>""", unsafe_allow_html=True)

with f3:
    st.markdown(
        f"""<div style="background:#f0f7ff;border-radius:8px;padding:16px 18px;height:100%;">
<div style="font-size:1.5rem;margin-bottom:6px;">🤖</div>
<div style="font-weight:700;margin-bottom:6px;">위험과 자원의 불일치</div>
<div style="font-size:0.88rem;color:#555;">
K-means 고위험 등급이면서 비상소화장치 밀도가
전북 중앙값 미만인 지역: <b>{', '.join(_gap_urgent) if _gap_urgent else '없음'}</b>.
위험도와 인프라 배치가 일치하지 않습니다.
</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 우선 지원 대상 ──────────────────────────────────────────
st.markdown("#### 우선 지원 대상")

_conclusion_df = summary_df[["시군", "우선순위", "cluster_label", "천호당화재율",
                              "고령단독주택거주율", "고령1인가구밀도"]].copy()
_conclusion_df = _conclusion_df.merge(
    gap_df[["시군", "갭_위험", "장치밀도_천호당"]], on="시군", how="left"
).sort_values("우선순위")
_conclusion_df["즉시보강"] = _conclusion_df["갭_위험"].map({True: "●", False: ""})
_conclusion_df = _conclusion_df.drop(columns=["갭_위험"]).rename(columns={
    "cluster_label": "ML 등급",
    "고령단독주택거주율": "고령단독거주율(%)",
    "고령1인가구밀도": "고령밀도(천호당)",
    "장치밀도_천호당": "장치밀도(개/천호)",
    "즉시보강": "장치 즉시보강",
})
st.dataframe(_conclusion_df.head(7), use_container_width=True, hide_index=True)
st.caption("장치 즉시보강 ● = ML 고위험 + 비상소화장치 밀도 중앙값 미만")

st.divider()

# ── 예방 전략 3가지 ──────────────────────────────────────────
st.markdown("#### 예방 전략")

p1, p2, p3 = st.columns(3)

with p1:
    st.markdown(
        """<div style="border:1.5px solid #e0e0e0;border-radius:8px;padding:16px 18px;height:100%;">
<div style="font-size:0.7rem;font-weight:700;color:#d62728;letter-spacing:0.1em;margin-bottom:8px;">POLICY 01</div>
<div style="font-weight:700;font-size:1rem;margin-bottom:8px;">찾아가는 전기안전 점검 우선 배정</div>
<div style="font-size:0.85rem;color:#555;">
우선순위 상위 시군의 고령 1인가구 단독주택을 대상으로
한국전기안전공사의 <b>무료 방문 점검</b>을 우선 배정.
경로당·복지관 연계 신청 접수 체계 구축.
</div></div>""", unsafe_allow_html=True)

with p2:
    st.markdown(
        f"""<div style="border:1.5px solid #e0e0e0;border-radius:8px;padding:16px 18px;height:100%;">
<div style="font-size:0.7rem;font-weight:700;color:#d62728;letter-spacing:0.1em;margin-bottom:8px;">POLICY 02</div>
<div style="font-weight:700;font-size:1rem;margin-bottom:8px;">비상소화장치 갭 지역 우선 보강</div>
<div style="font-size:0.85rem;color:#555;">
ML 고위험 + 장치밀도 부족 지역(<b>{', '.join(_gap_urgent) if _gap_urgent else '해당없음'}</b>)에
비상소화장치 추가 배치. 소방관서가 1~2개인
진안·임실·무주·순창은 대응 시간 보완책 병행 필요.
</div></div>""", unsafe_allow_html=True)

with p3:
    st.markdown(
        """<div style="border:1.5px solid #e0e0e0;border-radius:8px;padding:16px 18px;height:100%;">
<div style="font-size:0.7rem;font-weight:700;color:#d62728;letter-spacing:0.1em;margin-bottom:8px;">POLICY 03</div>
<div style="font-weight:700;font-size:1rem;margin-bottom:8px;">노후 배선 교체비 지원 확대</div>
<div style="font-size:0.85rem;color:#555;">
전기화재의 주원인인 절연열화·트래킹은
<b>노후 배선 교체</b>로 직접 예방 가능.
저소득 고령 1인가구 대상 배선·누전차단기 교체비
지원 사업 우선순위 상위 시군 집중 배정.
</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    "본 분석은 2021–2025년 한국전기안전공사·KOSIS 공공데이터를 기반으로 합니다. "
    "정책 효과는 사업 시행 후 데이터로 재검증이 필요합니다."
)
