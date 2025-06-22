#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import zipfile
from openai import OpenAI
import os
import fitz
import re


sns.set_theme(style='whitegrid', font_scale=1.5)
sns.set_palette('Set2', n_colors=10)
plt.rc('axes', unicode_minus=False)

import streamlit as st
from PIL import Image
from datetime import date

df = pd.read_csv('data/trans_raw.csv', encoding='euc-kr')
lat = pd.read_csv('data/lat.csv', encoding='euc-kr')
df['구매일자'] = pd.to_datetime(df['구매일자'])

st.set_page_config(page_title='auchelper', 
                   page_icon='🐔', layout='wide')

col1, col2 = st.columns([1, 4])
with col1:
    st.image("image/logo.PNG", width=150) #auc 로고
with col2:
    st.markdown(
        """
        <h1 style="margin-top: -12px;">안양도시공사 <strong>HELPER</strong></h1>
        """,
        unsafe_allow_html=True
    )

##################사이드바##################
st.sidebar.title("기능 선택")
menu = st.sidebar.radio(
    "원하는 기능을 선택하세요",
    ("📂 로바스 시각화(준비중)", "🖼️ 이미지 용량 줄이기", "🤖 안도미AI", "👥 준비중")
)

my_df = df

if menu == "📂 로바스 시각화(준비중)":
    st.sidebar.title("사이드 바 제목")
    st.sidebar.header("사이드 바 소제목")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("시작일시", date(2021, 1, 1),
                                        min_value=date(2021,1,1),
                                        max_value=date(2021,12,30))
    with col2:
        end_date = st.date_input("종료일시", date(2021, 12, 31),
                                        min_value=date(2021,1,2),
                                        max_value=date(2021,12,31))
    my_df = my_df[my_df['구매일자'].dt.date.between(start_date, end_date)]

    option01 = st.sidebar.checkbox('취소거래제외', value=False)
    if option01:
        my_df = my_df[my_df['취소여부']!=1]
         
    st.sidebar.header('상품분류선택')
    option02 = st.sidebar.multiselect('상품대분류', (my_df.상품대분류명.unique()), default=(my_df.상품대분류명.unique()))
    my_df = my_df[my_df.상품대분류명.isin(option02)]
    option03 = st.sidebar.multiselect('상품중분류', (my_df.상품중분류명.unique()), default=(my_df.상품중분류명.unique()))
    my_df = my_df[my_df.상품중분류명.isin(option03)]

##################메인##################

if menu == "📂 로바스 시각화(준비중)": 
    st.header("📂 로바스 자료 업로드")
    st.caption("★로바스에서 지출/어디어디/어디어디 엑셀로 자료를 다운받으신 후 업로드 해주세요")
    uploaded_file = st.file_uploader("엑셀 또는 CSV 파일 업로드", type=["xlsx", "xls", "csv"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                try:
                    df_uploaded = pd.read_csv(uploaded_file, encoding='utf-8')
                except:
                    df_uploaded = pd.read_csv(uploaded_file, encoding='euc-kr')
            else:
                df_uploaded = pd.read_excel(uploaded_file)

            st.success("파일 업로드 완료 ✅")
            st.dataframe(df_uploaded.head())
        except Exception as e:
            st.error(f"파일을 읽는 중 오류 발생: {e}")

    col1, col2, col3 = st.columns(3)
    col1.metric(label = "평균 판매액(단위:만원)", value = round(my_df['구매금액'].mean() / 10000,3), 
                delta=round(my_df['구매금액'].mean() / 10000 - df['구매금액'].mean() / 10000, 3))
    col2.metric(label = "구매 고객수", value = my_df['ID'].nunique(),
                delta=my_df['ID'].nunique() - df['ID'].nunique())
    col3.metric(label = "고객 평균 연령", value = round(my_df.groupby('ID')['연령'].mean().mean(),3),
                delta = round(my_df.groupby('ID')['연령'].mean().mean() - df.groupby('ID')['연령'].mean().mean(),3))


    st.header('1. 매출현황분석')

    st.subheader('전체')
    time_frame = st.selectbox("월별/주별/요일별", ("month", "week","weekday"))
    whole_values = my_df.groupby(time_frame)[['구매금액']].sum()
    whole_values.index.name = 'index'
    st.download_button('Download',whole_values.to_csv(encoding='euc-kr'), '매출현황분석.csv')
    st.area_chart(whole_values, use_container_width=True)


    st.subheader('지역별 비교')

    city_range = st.radio(label="범위선택", options=("시단위", "구단위"), index=0)

    if city_range=='시단위':
        city_range='구매지역_대분류'
        small_region=False
    else:
        city_range='구매지역_소분류'
        small_region = st.multiselect("구선택", (my_df.구매지역_소분류.unique()), (my_df.구매지역_소분류.unique()))

    if small_region==False:
        city_values = my_df
    else:
        city_values = my_df[my_df['구매지역_소분류'].isin(small_region)]
        
    city_values = pd.pivot_table(city_values, index=time_frame, columns=city_range, 
                                values='구매금액', aggfunc='sum',fill_value=0)
    city_values.index.name = None
    city_values.columns = list(city_values.columns)

    st.line_chart(city_values, use_container_width=True)

    st.subheader('Top5 비교')

    def top5(col_name, top=5):
        my_agg = (my_df.groupby(col_name)['구매금액'].sum()/1000000).reset_index().sort_values('구매금액', ascending=False).head(top)
        my_agg[col_name] = my_agg[col_name].astype('str')
        fig = plt.figure(figsize=(15,10))
        ax = sns.barplot(x='구매금액', y=col_name, data=my_agg)
        ax.bar_label(ax.containers[0], label_type='center', color='white')
        return fig

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('Top5 구매지역(단위:백만원)')
        st.pyplot(top5('구매지역_소분류'))
    with col2:
        st.write('Top5 구매시간(단위:백만원)')
        st.pyplot(top5('구매시간'))
    with col3:
        st.write('Top5 구매상품(단위:백만원)')
        st.pyplot(top5('상품중분류명'))


    st.header('2. 고객현황분석')


    st.subheader('성별 현황')
    st.write('성별 구매건수')
    gender_count = my_df.groupby([time_frame, '성별'])['구매수량'].sum().unstack()
    gender_count.columns = ['남성','여성']
    st.bar_chart(data=gender_count, use_container_width=True)


    st.subheader('연령분포')
    age_frame = st.selectbox("조건화선택", ("전체", "성별","취소여부","구매지역_대분류"))
    if age_frame=='전체':
        fig = sns.displot(x='연령', data=my_df, height=7, rug=True, kde=True)
    else:
        fig = sns.displot(x='연령', data=my_df, height=7, rug=True, hue=age_frame, kde=True)
    st.pyplot(fig)


    st.subheader('지역별분포')
    lat = lat.rename(columns={'지역':'거주지역'})
    map_lat = my_df[['거주지역']].merge(lat)
    jit = np.random.randn(len(map_lat), 2)
    jit_ratio = 0.01
    map_lat[['lat','lon']] = map_lat[['lat','lon']] + jit*jit_ratio
    st.map(map_lat)

if menu == "🖼️ 이미지 용량 줄이기":
    # 압축률 선택
    if "quality" not in st.session_state:
        st.session_state.quality = 60  # 기본값

    # 버튼 가운데 정렬 + 넓게 만들기
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        if st.button("20%", use_container_width=True):
            st.session_state.quality = 20
    with col2:
        if st.button("40%", use_container_width=True):
            st.session_state.quality = 40
    with col3:
        if st.button("60%", use_container_width=True):
            st.session_state.quality = 60
    with col4:
        if st.button("80%", use_container_width=True):
            st.session_state.quality = 80

    # 선택된 버튼 강조해서 시각적으로 보여주기
    compression_quality = st.session_state.quality
    st.markdown(
        f"<div style='text-align:center; font-weight:bold; font-size:18px;'>🔧 현재 선택된 압축률: <span style='color:#ff4b4b'>{compression_quality}%</span></div>",
        unsafe_allow_html=True
    )
    st.caption("※ 숫자가 낮을수록 이미지 크기가 작아집니다 (화질도 함께 낮아짐)")

    # 업로드 상태를 세션에 저장
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    # 파일 업로드
    new_files = st.file_uploader(
        "📂 이미지를 드래그 앤 드롭 하세요 (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="uploader"
    )

    # 새 업로드 발생 시 기존 목록 초기화
    if new_files:
        st.session_state.uploaded_files = new_files

    # 저장 및 다운로드 버튼
    if st.session_state.uploaded_files:
        if st.button("📉 이미지 용량 줄이기"):
            compressed_files = []
            for file in st.session_state.uploaded_files:
                image = Image.open(file)
                if image.mode in ("RGBA", "P"):
                    image = image.convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=compression_quality)
                buffer.seek(0)
                compressed_files.append((file.name, buffer))

            st.success(f"{len(compressed_files)}개의 이미지가 {compression_quality}% 품질로 압축되었습니다.")

            if len(compressed_files) == 1:
                name, buf = compressed_files[0]
                st.download_button(
                    label="📥 압축된 이미지 다운로드",
                    data=buf,
                    file_name=f"compressed_{name}",
                    mime="image/jpeg"
                )
            else:
                zip_io = io.BytesIO()
                with zipfile.ZipFile(zip_io, "w") as zf:
                    for name, buf in compressed_files:
                        zf.writestr(f"compressed_{name}", buf.getvalue())
                zip_io.seek(0)
                st.download_button(
                    label="📦 ZIP으로 다운로드",
                    data=zip_io,
                    file_name="compressed_images.zip",
                    mime="application/zip"
                )

if menu == "🤖 안도미AI": 
    # OpenAI API 클라이언트 초기화
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 시스템 역할 프롬프트
system_prompt = "너는 안양도시공사 규정집 전문가야. 사용자가 질문하면 해당 문서에서 관련 내용을 찾아 요약/설명해줘."

# PDF에서 문단 단위로 텍스트 추출
@st.cache_data
def load_pdf_chunks():
    with open("data/kj.pdf", "rb") as f:
        doc = fitz.open(stream=f.read(), filetype="pdf")
        all_text = ""
        for page in doc:
            all_text += page.get_text()
        # 문단 단위로 분리 (두 개 이상의 줄바꿈 또는 마침표 기준)
        chunks = re.split(r"\n{2,}|(?<=\.)\s+", all_text)
        # 너무 짧은 문단 제거
        chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50]
        return chunks

chunks = load_pdf_chunks()

# 유사 문단 검색 함수 (질문과 관련된 문단만 추출)
def find_relevant_chunks(chunks, question, top_n=5):
    keywords = set(re.findall(r"[\w가-힣]{2,}", question))
    scores = []
    for chunk in chunks:
        chunk_words = set(re.findall(r"[\w가-힣]{2,}", chunk))
        overlap = keywords & chunk_words
        score = len(overlap)
        scores.append((score, chunk))
    scores.sort(reverse=True, key=lambda x: x[0])
    return [chunk for _, chunk in scores[:top_n] if _ > 0]

# 사용자 질문 입력
user_input = st.text_area("질문을 입력하세요", placeholder="예: 연차휴가 규정이 어떻게 되나요?")

# GPT 호출
if st.button("🤖 GPT에게 질문") and user_input:
    with st.spinner("GPT가 관련 내용을 찾고 있습니다..."):
        relevant_chunks = find_relevant_chunks(chunks, user_input, top_n=5)
        if not relevant_chunks:
            st.warning("❗ 관련 문서를 찾을 수 없습니다. 더 구체적으로 질문해보세요.")
        else:
            context = "\n\n".join(relevant_chunks)
            prompt = f"다음은 규정 문서 일부입니다:\n{context}\n\n사용자 질문: {user_input}"
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = response.choices[0].message.content
                st.success("✅ GPT의 응답")
                st.markdown(answer)
            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")
