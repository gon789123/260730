"""
전국 시군구별 고령화율(65세 이상 인구 비율) 지도
- 인구 데이터: 읍·면·동 단위 연도별 인구 (계_나이 형식의 열 포함)
- 경계 데이터: 전국 시군구 255개 GeoJSON
- 지역 매칭은 이름이 아니라 '코드'(5자리 시군구 코드)로 수행
"""

import re
import json

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
# 옅은 색 -> 진한 색 순서 (ColorBrewer Reds 5단계)
BIN_COLORS = ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]


# ------------------------------------------------------------
# 데이터 불러오기 (캐시를 사용해서 매번 다시 받지 않도록 함)
# ------------------------------------------------------------
@st.cache_data(show_spinner="인구 데이터를 불러오는 중...")
def load_population(url: str) -> pd.DataFrame:
    """읍·면·동 단위 인구 데이터를 읽어온다.

    '코드' 열은 계산용 숫자가 아니라 이름표이므로 반드시 문자열(str)로 읽는다.
    """
    df = pd.read_csv(url, compression="gzip", dtype={"코드": str})
    # 혹시 코드 앞의 0이 사라졌을 경우를 대비해 문자열 처리를 한 번 더 확실히 함
    df["코드"] = df["코드"].astype(str)
    return df


@st.cache_data(show_spinner="지도 경계 데이터를 불러오는 중...")
def load_geojson(url: str) -> dict:
    """전국 시군구 경계 GeoJSON을 읽어온다."""
    resp = requests.get(url)
    resp.raise_for_status()
    geo = resp.json()

    # geojson의 '코드' 속성도 문자열인지 확인(자리수 통일)
    for feature in geo["features"]:
        props = feature["properties"]
        props["코드"] = str(props["코드"])
    return geo


def get_age_columns(df: pd.DataFrame):
    """'계_0세' ~ '계_100세 이상' 형식의 나이별(남녀 합계) 열 이름과 나이를 뽑아낸다."""
    age_cols = []
    for col in df.columns:
        if not col.startswith("계_"):
            continue
        m = re.match(r"^계_(\d+)세(\s*이상)?$", col)
        if m:
            age = int(m.group(1))
            age_cols.append((col, age))
    return age_cols


@st.cache_data(show_spinner="고령화율을 계산하는 중...")
def build_sigungu_ratio(pop_df: pd.DataFrame) -> pd.DataFrame:
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

    # 4) 전체 인구, 65세 이상 인구를 각 읍면동 행에 대해 계산
    df["전체인구"] = df[all_age_col_names].sum(axis=1)
    df["고령인구"] = df[old_col_names].sum(axis=1)

    # 5) 시군구 단위로 합산
    grouped = (
        df.groupby("시군구코드", as_index=False)
        .agg(
            전체인구=("전체인구", "sum"),
            고령인구=("고령인구", "sum"),
            시도=("시도", "first"),
            시군구=("시군구", "first"),
        )
    )

    # 6) 고령화율(%) 계산
    grouped["고령화율"] = (grouped["고령인구"] / grouped["전체인구"] * 100).round(2)

    # 7) 5단계 구간으로 나누기
    grouped["구간"] = pd.cut(
        grouped["고령화율"], bins=BIN_EDGES, labels=range(len(BIN_LABELS))
    ).astype(int)

    return grouped, latest_year


# ------------------------------------------------------------
# 데이터 준비
# ------------------------------------------------------------
pop_df = load_population(POP_URL)
geojson_data = load_geojson(GEO_URL)
ratio_df, latest_year = build_sigungu_ratio(pop_df)

st.title("🗺️ 전국 시군구별 고령화 지도")
st.caption(f"기준 연도: {latest_year}년 · 65세 이상 인구 비율(고령화율)")

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
# 고령화율 상위 10곳 / 하위 10곳 표
# ------------------------------------------------------------
st.subheader("고령화율 순위")

top10 = ratio_df.sort_values("고령화율", ascending=False).head(10)
bottom10 = ratio_df.sort_values("고령화율", ascending=True).head(10)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🔺 고령화율이 높은 시군구 TOP 10**")
    st.dataframe(
        top10[["시도", "시군구", "고령화율"]].reset_index(drop=True),
        use_container_width=True,
    )

with col2:
    st.markdown("**🔻 고령화율이 낮은 시군구 TOP 10**")
    st.dataframe(
        bottom10[["시도", "시군구", "고령화율"]].reset_index(drop=True),
        use_container_width=True,
    )
