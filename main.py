"""
전국 시군구별 고령화율(65세 이상 인구 비율) 지도
- 인구 데이터: 읍·면·동 단위 연도별 인구 (계_나이 형식의 열 포함)
- 경계 데이터: 전국 시군구 255개 GeoJSON
- 지역 매칭은 이름이 아니라 '코드'(5자리 시군구 코드)로 수행
"""

import re
import json
import io

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
st.set_page_config(page_title="전국 고령화 지도", layout="wide")

POP_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/population_yearly.csv.gz"
GEO_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/boundaries/sigungu_kr.geojson"

# 5단계 구간 경계값 (전국 시군구를 다섯 덩어리로 나눈 실제 값)
BIN_EDGES = [-np.inf, 19, 23, 28, 38, np.inf]
BIN_LABELS = ["19% 미만", "19~23%", "23~28%", "28~38%", "38% 이상"]
# 옅은 색 -> 진한 색 순서 (ColorBrewer Blues 5단계)
BIN_COLORS = ["#eff3ff", "#bdd7e7", "#6baed6", "#3182bd", "#08519c"]

# 고령화 단계 기준선 (국제/통계청 통용 기준): 65세 이상 인구 비율 기준
AGING_STAGES = [
    (7, "고령화사회"),
    (14, "고령사회"),
    (20, "초고령사회"),
]


# ------------------------------------------------------------
# 데이터 불러오기 (캐시를 사용해서 매번 다시 받지 않도록 함)
# ------------------------------------------------------------
@st.cache_data(show_spinner="인구 데이터를 불러오는 중...")
def load_population(url: str) -> pd.DataFrame:
    """읍·면·동 단위 인구 데이터를 읽어온다.

    '코드' 열은 계산용 숫자가 아니라 이름표이므로 반드시 문자열(str)로 읽는다.
    네트워크가 느리거나 실패할 때 앱이 원인 모르게 멈추지 않도록,
    타임아웃을 두고 실패하면 화면에 에러 메시지를 보여준다.
    """
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        st.error(f"인구 데이터를 받아오지 못했어요. 잠시 후 다시 시도해 주세요. ({e})")
        st.stop()

    df = pd.read_csv(io.BytesIO(resp.content), compression="gzip", dtype={"코드": str})
    # 혹시 코드 앞의 0이 사라졌을 경우를 대비해 문자열 처리를 한 번 더 확실히 함
    df["코드"] = df["코드"].astype(str)
    return df


@st.cache_data(show_spinner="지도 경계 데이터를 불러오는 중...")
def load_geojson(url: str) -> dict:
    """전국 시군구 경계 GeoJSON을 읽어온다."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        st.error(f"지도 경계 데이터를 받아오지 못했어요. 잠시 후 다시 시도해 주세요. ({e})")
        st.stop()

    geo = resp.json()

    # geojson의 '코드' 속성도 문자열인지 확인(자리수 통일)
    for feature in geo["features"]:
        props = feature["properties"]
        props["코드"] = str(props["코드"])
    return geo


@st.cache_data(show_spinner="지도 기준 지역 이름을 정리하는 중...")
def build_geo_names(_geojson_data: dict) -> pd.DataFrame:
    """geojson에 있는 코드별 정확한 시도·시군구 이름을 가져온다.

    인구 데이터의 '시군구' 열은 수원시·화성시·창원시 같은 대도시의 경우
    일반구(예: 효행구, 동탄구) 구분 없이 그냥 '화성시'로만 되어 있어서,
    코드는 다른데 이름이 똑같은 지역이 여럿 생긴다. 반면 geojson은 각 코드마다
    '화성시효행구'처럼 구까지 정확히 구분된 이름을 갖고 있으므로, 화면에 보여줄
    이름은 population 데이터가 아니라 geojson 쪽 이름을 기준으로 삼는다.
    """
    records = [
        {
            "시군구코드": f["properties"]["코드"],
            "시도": f["properties"]["시도"],
            "시군구": f["properties"]["시군구"],
        }
        for f in _geojson_data["features"]
    ]
    return pd.DataFrame(records)


def get_age_columns(df: pd.DataFrame, prefix: str = "계_"):
    """'계_0세'~'계_100세 이상' 같은 나이별 열 이름과 나이를 뽑아낸다.

    prefix를 '남_' 또는 '여_'로 바꾸면 성별 인구 열도 같은 방식으로 뽑을 수 있다.
    """
    age_cols = []
    for col in df.columns:
        if not col.startswith(prefix):
            continue
        m = re.match(rf"^{re.escape(prefix)}(\d+)세(\s*이상)?$", col)
        if m:
            age = int(m.group(1))
            age_cols.append((col, age))
    return age_cols


@st.cache_data(show_spinner="고령화율을 계산하는 중...")
def build_sigungu_ratio(pop_df: pd.DataFrame, geo_names: pd.DataFrame) -> pd.DataFrame:
    """읍면동 인구 데이터를 시군구 단위로 묶어서 65세 이상 인구 비율을 계산한다."""

    # 1) 가장 최신 연도만 사용
    latest_year = pop_df["연도"].max()
    df = pop_df[pop_df["연도"] == latest_year].copy()

    # 2) '코드' 앞 5자리 = 시군구 코드
    df["시군구코드"] = df["코드"].str[:5]

    # 3) 나이별(계_) 열 목록과 나이 추출
    age_cols = get_age_columns(df)
    all_age_col_names = [c for c, _ in age_cols]
    old_col_names = [c for c, age in age_cols if age >= 65]
    working_col_names = [c for c, age in age_cols if 15 <= age <= 64]

    # 4) 전체 인구, 65세 이상 인구, 생산연령인구(15~64세)를 각 읍면동 행에 대해 계산
    df["전체인구"] = df[all_age_col_names].sum(axis=1)
    df["고령인구"] = df[old_col_names].sum(axis=1)
    df["생산연령인구"] = df[working_col_names].sum(axis=1)

    # 5) 시군구 단위로 인구만 합산 (이름은 population 데이터가 아니라 geo_names 사용)
    grouped = (
        df.groupby("시군구코드", as_index=False)
        .agg(
            전체인구=("전체인구", "sum"),
            고령인구=("고령인구", "sum"),
            생산연령인구=("생산연령인구", "sum"),
        )
    )

    # 5-1) 수원시·화성시·창원시처럼 일반구가 있는 대도시는 population 데이터의
    #      '시군구' 열에 구 이름이 빠져 있어서(예: '화성시'만 4번 반복) 지도(geojson)
    #      쪽의 정확한 이름('화성시효행구' 등)으로 덮어쓴다.
    grouped = grouped.merge(geo_names, on="시군구코드", how="left")

    # 6) 고령화율(%) 계산
    grouped["고령화율"] = (grouped["고령인구"] / grouped["전체인구"] * 100).round(2)

    # 6-1) 노년부양비(%) = 고령인구 / 생산연령인구(15~64세) * 100
    #      생산연령인구 100명이 고령자를 몇 명 부양하는지 나타내는 지표
    grouped["노년부양비"] = (
        grouped["고령인구"] / grouped["생산연령인구"] * 100
    ).round(1)

    # 6-2) 부양인원 = 생산연령인구 / 고령인구 -> "생산인구 몇 명당 고령자 1명"
    grouped["부양인원"] = (grouped["생산연령인구"] / grouped["고령인구"]).round(1)

    # 7) 5단계 구간으로 나누기
    grouped["구간"] = pd.cut(
        grouped["고령화율"], bins=BIN_EDGES, labels=range(len(BIN_LABELS))
    ).astype(int)

    return grouped, latest_year


@st.cache_data(show_spinner="연도별 고령화율을 계산하는 중...")
def build_all_years_ratio(pop_df: pd.DataFrame, geo_names: pd.DataFrame) -> pd.DataFrame:
    """모든 연도에 대해 시군구별 고령화율을 계산한다 (애니메이션 지도용)."""

    df = pop_df.copy()
    df["시군구코드"] = df["코드"].str[:5]

    age_cols = get_age_columns(df, "계_")
    all_age_col_names = [c for c, _ in age_cols]
    old_col_names = [c for c, age in age_cols if age >= 65]

    df["전체인구"] = df[all_age_col_names].sum(axis=1)
    df["고령인구"] = df[old_col_names].sum(axis=1)

    grouped = (
        df.groupby(["연도", "시군구코드"], as_index=False)
        .agg(
            전체인구=("전체인구", "sum"),
            고령인구=("고령인구", "sum"),
        )
    )

    # population 데이터의 '시군구' 열 대신 geojson 쪽 정확한 이름을 붙인다
    grouped = grouped.merge(geo_names, on="시군구코드", how="left")

    grouped["고령화율"] = (grouped["고령인구"] / grouped["전체인구"] * 100).round(2)
    grouped["구간"] = pd.cut(
        grouped["고령화율"], bins=BIN_EDGES, labels=range(len(BIN_LABELS))
    ).astype(int)
    grouped["구간라벨"] = grouped["구간"].map(lambda i: BIN_LABELS[i])

    return grouped


@st.cache_data(show_spinner="인구 피라미드를 만드는 중...")
def build_pyramid(pop_df: pd.DataFrame, year: int, sigungu_code: str) -> pd.DataFrame:
    """선택한 연도·시군구의 5세 단위 남녀 인구 피라미드 데이터를 만든다."""

    df = pop_df[pop_df["연도"] == year].copy()
    df["시군구코드"] = df["코드"].str[:5]
    df = df[df["시군구코드"] == sigungu_code]

    male_cols = get_age_columns(df, "남_")
    female_cols = get_age_columns(df, "여_")

    def age_bin(age: int) -> str:
        """나이를 5세 단위 구간 이름으로 바꾼다 (예: 23 -> '20~24세')."""
        if age >= 100:
            return "100세 이상"
        start = (age // 5) * 5
        return f"{start}~{start + 4}세"

    # 나이 구간 순서 (0~4세부터 100세 이상까지, 어린 나이가 먼저 오도록)
    bin_order = [f"{i}~{i + 4}세" for i in range(0, 100, 5)] + ["100세 이상"]

    male_sum = {b: 0 for b in bin_order}
    for col, age in male_cols:
        male_sum[age_bin(age)] += df[col].sum()

    female_sum = {b: 0 for b in bin_order}
    for col, age in female_cols:
        female_sum[age_bin(age)] += df[col].sum()

    pyramid_df = pd.DataFrame(
        {
            "연령대": bin_order,
            "남": [male_sum[b] for b in bin_order],
            "여": [female_sum[b] for b in bin_order],
        }
    )
    return pyramid_df


# ------------------------------------------------------------
# 데이터 준비
# ------------------------------------------------------------
pop_df = load_population(POP_URL)
geojson_data = load_geojson(GEO_URL)
geo_names = build_geo_names(geojson_data)
ratio_df, latest_year = build_sigungu_ratio(pop_df, geo_names)

st.title("🗺️ 전국 시군구별 고령화 지도")
st.caption(f"기준 연도: {latest_year}년 · 65세 이상 인구 비율(고령화율)")

with st.expander("📚 용어 설명: 고령화사회 · 고령사회 · 초고령사회란?"):
    st.markdown(
        """
65세 이상 인구 비율을 기준으로 사회를 아래 세 단계로 나눕니다.

| 단계 | 65세 이상 비율 | 의미 |
|---|---|---|
| 고령화사회 | 7% 이상 | 고령 인구가 늘어나기 시작하는 단계 |
| 고령사회 | 14% 이상 | 전체 인구 7명 중 1명이 고령자인 단계 |
| 초고령사회 | 20% 이상 | 전체 인구 5명 중 1명이 고령자인 단계 |

우리나라는 전국 평균으로 이미 초고령사회에 진입했으며, 아래 그래프의 점선이 이 세 기준선을 나타냅니다.

**노년부양비**란 생산연령인구(15~64세) 100명이 부양해야 하는 고령인구(65세 이상) 수를 뜻합니다.
숫자가 클수록 젊은 세대의 부양 부담이 크다는 의미입니다.
        """
    )

# ------------------------------------------------------------
# 지도 그리기 (단계구분도, 5단계 색, 배경 타일 없음)
# ------------------------------------------------------------
# 마우스 오버 시 보여줄 텍스트
ratio_df["hover_text"] = (
    ratio_df["시도"] + " " + ratio_df["시군구"]
    + "<br>고령화율: " + ratio_df["고령화율"].astype(str) + "%"
)

# 구간(0~4)을 색 5개로 나눈 이산(discrete) 컬러스케일 만들기
n_bins = len(BIN_LABELS)
colorscale = []
for i, color in enumerate(BIN_COLORS):
    colorscale.append([i / n_bins, color])
    colorscale.append([(i + 1) / n_bins, color])

# z값은 구간의 가운데 값(0.5, 1.5, ...)으로 넣어서 경계선에서 색이 헷갈리지 않게 함
z_values = ratio_df["구간"] + 0.5

fig = go.Figure(
    go.Choropleth(
        geojson=geojson_data,
        locations=ratio_df["시군구코드"],
        z=z_values,
        featureidkey="properties.코드",
        text=ratio_df["hover_text"],
        hoverinfo="text",
        colorscale=colorscale,
        zmin=0,
        zmax=n_bins,
        marker_line_color="white",
        marker_line_width=0.5,
        colorbar=dict(
            title="고령화율",
            tickmode="array",
            tickvals=[i + 0.5 for i in range(n_bins)],
            ticktext=BIN_LABELS,
        ),
    )
)

# 배경 지도 타일 없이 경계선만 보이도록 설정
fig.update_geos(
    fitbounds="locations",
    visible=False,
    showcountries=False,
    showcoastlines=False,
    showland=False,
    showframe=False,
    bgcolor="rgba(0,0,0,0)",
)

fig.update_layout(
    margin=dict(l=0, r=0, t=10, b=0),
    height=700,
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# 고령화율 순위 막대그래프
# ------------------------------------------------------------
st.subheader("고령화율 순위 그래프")

# 전국 평균(인구가중 평균) 고령화율 계산
national_avg = ratio_df["고령인구"].sum() / ratio_df["전체인구"].sum() * 100

rank_option = st.radio(
    "표시할 범위를 골라주세요",
    ["상위 15개", "하위 15개", "상위 15 + 하위 15"],
    horizontal=True,
)

if rank_option == "상위 15개":
    rank_df = ratio_df.sort_values("고령화율", ascending=False).head(15)
elif rank_option == "하위 15개":
    rank_df = ratio_df.sort_values("고령화율", ascending=True).head(15)
else:
    top15 = ratio_df.sort_values("고령화율", ascending=False).head(15)
    low15 = ratio_df.sort_values("고령화율", ascending=True).head(15)
    rank_df = pd.concat([top15, low15]).drop_duplicates(subset="시군구코드")

# 값이 작은 것이 아래에, 큰 것이 위에 오도록 정렬 (가로 막대그래프 특성상 역순 정렬)
rank_df = rank_df.sort_values("고령화율", ascending=True)
rank_df["지역명"] = rank_df["시도"] + " " + rank_df["시군구"]

# 지도와 같은 5단계 색을 막대그래프에도 그대로 사용해서 통일감을 줌
bar_colors = rank_df["구간"].map(lambda i: BIN_COLORS[i])

bar_fig = go.Figure(
    go.Bar(
        x=rank_df["고령화율"],
        y=rank_df["지역명"],
        orientation="h",
        marker_color=bar_colors,
        text=rank_df["고령화율"].astype(str) + "%",
        textposition="outside",
    )
)

# 전국 평균선을 점선으로 표시
bar_fig.add_vline(
    x=national_avg,
    line_dash="dash",
    line_color="gray",
    annotation_text=f"전국 평균 {national_avg:.1f}%",
    annotation_position="top",
)

# 고령화사회·고령사회·초고령사회 기준선(7%·14%·20%) 표시
for stage_value, stage_name in AGING_STAGES:
    bar_fig.add_vline(
        x=stage_value,
        line_dash="dot",
        line_color="darkorange",
        annotation_text=f"{stage_name} {stage_value}%",
        annotation_position="bottom",
        annotation_font_color="darkorange",
    )

bar_fig.update_layout(
    xaxis_title="고령화율(%)",
    yaxis_title="",
    height=max(400, 24 * len(rank_df)),
    margin=dict(l=0, r=40, t=40, b=0),
)

st.plotly_chart(bar_fig, use_container_width=True)

# ------------------------------------------------------------
# 연도별 고령화 변화 지도 (슬라이더로 한 해씩 보기)
# ------------------------------------------------------------
st.subheader("연도별 고령화 변화 지도")
st.caption("슬라이더를 움직여서 연도별 변화를 볼 수 있어요.")

all_years_df = build_all_years_ratio(pop_df, geo_names)

year_list = sorted(all_years_df["연도"].unique())
selected_year = st.select_slider(
    "연도를 선택하세요", options=year_list, value=year_list[-1]
)

# 선택한 연도의 데이터만 뽑아서 지도 1장만 그린다
# (애니메이션으로 모든 연도를 한 번에 담으면 데이터 용량이 커져서
#  Streamlit Cloud에서 느려지거나 멈추는 문제가 있었기 때문에,
#  슬라이더로 한 해씩만 가볍게 그리는 방식으로 바꿈)
year_df = all_years_df[all_years_df["연도"] == selected_year].copy()
year_df["hover_text"] = (
    year_df["시도"] + " " + year_df["시군구"]
    + "<br>고령화율: " + year_df["고령화율"].astype(str) + "%"
)
year_z_values = year_df["구간"] + 0.5

year_fig = go.Figure(
    go.Choropleth(
        geojson=geojson_data,
        locations=year_df["시군구코드"],
        z=year_z_values,
        featureidkey="properties.코드",
        text=year_df["hover_text"],
        hoverinfo="text",
        colorscale=colorscale,
        zmin=0,
        zmax=n_bins,
        marker_line_color="white",
        marker_line_width=0.5,
        colorbar=dict(
            title="고령화율",
            tickmode="array",
            tickvals=[i + 0.5 for i in range(n_bins)],
            ticktext=BIN_LABELS,
        ),
    )
)

year_fig.update_geos(
    fitbounds="locations",
    visible=False,
    showcountries=False,
    showcoastlines=False,
    showland=False,
    showframe=False,
    bgcolor="rgba(0,0,0,0)",
)

year_fig.update_layout(
    margin=dict(l=0, r=0, t=10, b=0),
    height=700,
)

st.plotly_chart(year_fig, use_container_width=True)

# ------------------------------------------------------------
# 고령화율 상위 10곳 / 하위 10곳 표
# ------------------------------------------------------------
st.subheader("고령화율 순위 표")

top10 = ratio_df.sort_values("고령화율", ascending=False).head(10)
bottom10 = ratio_df.sort_values("고령화율", ascending=True).head(10)


def render_centered_table(table_df: pd.DataFrame):
    """가운데 정렬이 확실히 적용되는 HTML 표를 그린다.

    st.dataframe은 내부적으로 자체 그리드 컴포넌트를 쓰기 때문에
    pandas Styler로 지정한 text-align 같은 CSS가 반영되지 않는 경우가 있다.
    그래서 직접 HTML 표를 만들어 st.markdown으로 그리는 방식을 쓴다.
    """
    display_df = table_df.copy()
    display_df.index = range(1, len(display_df) + 1)
    display_df["고령화율"] = display_df["고령화율"].map(lambda v: f"{v:.2f}%")

    html_table = display_df.to_html(classes="ranking-table", border=0)

    styled_html = f"""
    <style>
    .ranking-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }}
    .ranking-table th, .ranking-table td {{
        text-align: center !important;
        padding: 6px 8px;
    }}
    .ranking-table thead th {{
        background-color: #3182bd;
        color: white;
    }}
    .ranking-table tbody tr:nth-child(even) {{
        background-color: #f2f6fc;
    }}
    </style>
    {html_table}
    """
    st.markdown(styled_html, unsafe_allow_html=True)


col1, col2 = st.columns(2)

with col1:
    st.markdown("**🔺 고령화율이 높은 시군구 TOP 10**")
    render_centered_table(top10[["시도", "시군구", "고령화율"]])

with col2:
    st.markdown("**🔻 고령화율이 낮은 시군구 TOP 10**")
    render_centered_table(bottom10[["시도", "시군구", "고령화율"]])

# ------------------------------------------------------------
# 지역별 상세 정보 (노년부양비 계산기 + 인구 피라미드)
# ------------------------------------------------------------
st.subheader("지역별 상세 정보")

# '도 선택' -> '시군구 선택' 2단계 선택창 (한 번에 다 찾는 것보다 가독성이 좋음)
region_df = ratio_df.copy()

sido_list = sorted(region_df["시도"].unique())

sel_col1, sel_col2 = st.columns(2)

with sel_col1:
    selected_sido = st.selectbox("① 시·도를 선택하세요", sido_list)

sigungu_list = sorted(
    region_df.loc[region_df["시도"] == selected_sido, "시군구"].unique()
)

with sel_col2:
    selected_sigungu = st.selectbox("② 시·군·구를 선택하세요", sigungu_list)

selected_row = region_df[
    (region_df["시도"] == selected_sido) & (region_df["시군구"] == selected_sigungu)
].iloc[0]
selected_code = selected_row["시군구코드"]
selected_label = f"{selected_sido} {selected_sigungu}"

# --- 노년부양비 계산기 ---
st.markdown(f"#### 🧮 {selected_label} 노년부양비 계산기")

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric("고령화율 (65세 이상 비율)", f"{selected_row['고령화율']:.1f}%")

with metric_col2:
    st.metric("노년부양비", f"{selected_row['노년부양비']:.1f}%")

with metric_col3:
    st.metric("부양 구조", f"생산인구 {selected_row['부양인원']:.1f}명당 1명")

st.caption(
    "노년부양비 = 고령인구(65세 이상) ÷ 생산연령인구(15~64세) × 100. "
    "생산연령인구 100명이 고령자를 몇 명 부양해야 하는지를 나타냅니다."
)

# --- 인구 피라미드 ---
st.markdown("#### 👨‍👩‍👧‍👦 인구 피라미드")

pyramid_df = build_pyramid(pop_df, latest_year, selected_code)

pyramid_fig = go.Figure()

# 남자는 왼쪽으로 보이도록 값을 음수로 바꿔서 그림
pyramid_fig.add_trace(
    go.Bar(
        y=pyramid_df["연령대"],
        x=-pyramid_df["남"],
        name="남",
        orientation="h",
        marker_color="#3182bd",
    )
)
pyramid_fig.add_trace(
    go.Bar(
        y=pyramid_df["연령대"],
        x=pyramid_df["여"],
        name="여",
        orientation="h",
        marker_color="#de2d26",
    )
)

# x축 눈금이 음수로 보이지 않고 절댓값(실제 인구 수)으로 보이도록 설정
max_val = max(1, int(max(pyramid_df["남"].max(), pyramid_df["여"].max())))
tick_step = max(1, max_val // 4)
tick_vals = list(range(-tick_step * 4, tick_step * 4 + 1, tick_step))
tick_text = [str(abs(v)) for v in tick_vals]

pyramid_fig.update_layout(
    title=f"{selected_label} 인구 피라미드 ({latest_year}년)",
    barmode="overlay",
    bargap=0.1,
    xaxis_title="인구 수(명)",
    yaxis_title="연령대",
    yaxis=dict(categoryorder="array", categoryarray=pyramid_df["연령대"].tolist()),
    xaxis=dict(tickvals=tick_vals, ticktext=tick_text),
    height=800,
    margin=dict(l=0, r=0, t=40, b=0),
)

st.plotly_chart(pyramid_fig, use_container_width=True)
