import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

st.set_page_config(page_title="박스오피스 대시보드", layout="wide")
st.title("🎬 어제의 박스오피스")

# 비밀 금고에서 인증키 꺼내기 (코드에는 키를 적지 않는다)
KOBIS_KEY = st.secrets["KOBIS_KEY"]

# 한국 시간 기준 어제 날짜를 여덟 자리로 (배포 서버 시계는 외국 기준일 수 있다)
yesterday = datetime.now(ZoneInfo("Asia/Seoul")) - timedelta(days=1)
target_dt = yesterday.strftime("%Y%m%d")
st.caption(f"조회 기준일(어제): {yesterday.strftime('%Y-%m-%d')}")

url = "https://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json"
res = requests.get(url, params={"key": KOBIS_KEY, "targetDt": target_dt}, timeout=10)

if res.status_code != 200:
    st.error(f"요청이 실패했습니다 (상태코드: {res.status_code})")
    st.stop()

data = res.json()

# KOBIS는 키가 틀려도 상태코드 200을 준다. 대신 faultInfo 상자가 온다.
if "faultInfo" in data:
    st.error("인증키가 올바르지 않습니다. 금고(Secrets)의 KOBIS_KEY를 확인해 주세요.")
    st.stop()

box_list = data.get("boxOfficeResult", {}).get("dailyBoxOfficeList", [])
if not box_list:
    st.warning("그날 자료가 없습니다. 날짜를 하루 더 앞으로 옮겨 보세요.")
    st.stop()

df = pd.DataFrame(box_list)

# 글자로 온 숫자들을 진짜 숫자로 바꾸기
for col in ["rank", "audiCnt", "audiAcc", "scrnCnt", "showCnt"]:
    df[col] = pd.to_numeric(df[col])

# 1위 영화 지표 카드 세 장
top = df.sort_values("rank").iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("어제 1위", top["movieNm"])
c2.metric("어제 관객수", f"{top['audiCnt']:,}명")
c3.metric("누적 관객", f"{top['audiAcc']:,}명")

# 표를 한국어 열 이름으로 정리
table = df[["rank", "movieNm", "openDt", "audiCnt", "audiAcc", "scrnCnt"]].copy()
table.columns = ["순위", "영화명", "개봉일", "관객수", "누적관객", "스크린수"]
table = table.sort_values("순위").reset_index(drop=True)

st.subheader("📋 박스오피스 TOP 10")
st.dataframe(table)

st.subheader("📈 관객수 상위 5편")
top5 = table.sort_values("관객수", ascending=False).head(5)
st.bar_chart(top5.set_index("영화명")["관객수"])
