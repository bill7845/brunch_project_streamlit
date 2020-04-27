import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pickle
import sqlite3
import os
from konlpy.tag import Okt

import sklearn
import konlpy
import matplotlib as mat
import seaborn as sns
import plotly as plt


# print(mat.__version__)
# print(sns.__version__)
# print(plt.__version__)


## load csv
# @st.cache
def load_data():
    df = pd.read_csv('C:/Users/KIHyuk/Documents/GitHub/brunch_project_streamlit/pkl_objects/all_df.csv'
    ,index_col='Unnamed: 0')
    return df

## load count_vect
def load_keyword_count_vect():
    cur_dir = os.path.dirname(__file__)
    keyword_count_vect = pickle.load(open(os.path.join(cur_dir,'pkl_objects','keyword_count_vect.pkl'), 'rb'))
    return keyword_count_vect

## load keyword_mat
def load_keyword_mat():
    cur_dir = os.path.dirname(__file__)
    keyword_mat = pickle.load(open(os.path.join(cur_dir,'pkl_objects','keyword_mat.pkl'), 'rb'))
    return keyword_mat

## load classifier
def load_clf():
    cur_dir = os.path.dirname(__file__)
    clf = pickle.load(open(os.path.join(cur_dir,'pkl_objects','classifier.pkl'), 'rb'))
    return clf

## 형태소 분석기
okt = Okt()
def okt_tokenizer(text):
    tokens_ko = okt.morphs(text,stem=True)
    return tokens_ko

## laod tfidf vector
def load_tfidf_matrix():
    cur_dir = os.path.dirname(__file__)
    tfidf_matrix_train = pickle.load(open(os.path.join(cur_dir,'pkl_objects','tfidf_matrix_train.pkl'), 'rb'))
    return tfidf_matrix_train

# 입력받은 문서를 분류하여 분류결과,분류확률 반환
@st.cache # 분류기 한번 학습시 계속 결과 저장
def classify(document):
    label = {0:'지구한바퀴 세계여행', 1:'그림 웹툰', 2:'시사 이슈', 3:'IT 트렌드',4:'사진 촬영', 5:'취향저격 영화리뷰', 6:'뮤직 인사이드',
     7:'육아 이야기', 8:'요리 레시피',
    9:'건강 운동', 10:'멘탈관리 심리탐구', 11:'문화 예술', 12:'건축 설계',
    13:'인문학 철학',14:'쉽게 읽는 역사', 15:'우리집 반려동물', 16:'오늘은 이런 책',
    17:'직장인 현실 조언', 18:'디자인 스토리',19:'감성 에세이'}

    tfidf_matrix_train = load_tfidf_matrix()
    clf = load_clf()

    X = tfidf_matrix_train.transform([document])
    y = clf.predict(X)[0]

    proba = clf.predict_proba(X)
    proba_max = np.max(proba)

    return label[y],proba_max

## category에 해당하는 keyword 목록 반환
def get_categories(label,dict):
    return tuple(dict[label]) # multiselect box에서 사용위해 tuple로 반환

## 추천 시스템
def find_sim_document(df, count_vect, keyword_mat, input_keywords, top_n=10):
  input_keywords_mat = count_vect.transform(pd.Series(input_keywords))

  keyword_sim = cosine_similarity(input_keywords_mat, keyword_mat)

  keyword_sim_sorted_ind = keyword_sim.argsort()[:,::-1]

  top_n_sim = keyword_sim_sorted_ind[:1,:(top_n)]
  top_n_sim = top_n_sim.reshape(-1)

  return df.iloc[top_n_sim][['text','keyword']]

## keyword trend 차트
def keyword_trend_chart(df, select_keyword):
    df.index = pd.to_datetime(df['publish_date'],format='%Y-%m-%d')
    df = df['keyword']['2020-01-01':].resample('M').sum()

    res_df = pd.DataFrame(columns=select_keyword,index=df.index)
    for keyword in select_keyword:
        keyword_week_count = []
        for week in range(len(df)):
            keyword_week_count.append(df.iloc[week].count(keyword))
        res_df[keyword] = keyword_week_count

    return res_df

## sqlite query
def sqlite_main(document, answer, pred_label, correction_label, keyword_select):
    conn = sqlite3.connect('brunch_network.db')
    c = conn.cursor()
    c.execute("INSERT INTO main_table(text, answer, pred_label, correction_label, keyword_select, date)"\
    " VALUES (?, ?, ?, ?, ?, DATETIME('now'))", (document, answer, pred_label, correction_label, keyword_select))
    conn.commit()
    conn.close()

## main 함수
def main():

    st.sidebar.title("Menu")
    app_mode = st.sidebar.selectbox("",
        ["Home", "App 실행", "전체 Code"])

    ## 개요 페이지. (시작 페이지)
    if app_mode == "Home":

        st.title("Brunch Networking")
        st.subheader("부제목")
        st.write('---')

        st.write(
        '''
        ## < Content >

        * 브런치라는 서비스
        * 작가도 독자중의 한명
        * 머신러닝을 활용한 브런치 네트워킹
        '''
        )
        st.write("---")

        st.markdown(
        '''
        ## 1. 브런치라는 서비스



        '''
        )






    ## app 실행 페이지.
    elif app_mode == "App 실행":
        st.sidebar.success('앱 실행중입니다')
        st.title("환영합니다 작가님!")
        st.write("")
        st.write("")

        document = st.text_area("작성하신 글을 입력해주세요.") ## text 입력란

        submit_button = st.button("제출",key='document') # submit 버튼

        #######################################################################
        ## 1. 문서 입력 후 submit 버튼 클릭 시 분류 모델에 의해 분류라벨,확률값 출력
        #######################################################################
        if submit_button:
            label,proba_max = classify(document)
            st.write('작성하신 글은 %d퍼센트의 확률로 \'%s\' 카테고리로 분류됩니다.' %(round((proba_max)*100),label))

        #######################################################################
        ## 2. 분류 결과에 대한 맞춤,틀림 여부 입력받음
        ##      2.1 정답일 경우
        ##      2.2 오답일 경우
        #######################################################################
        category_list = ['<select>','지구한바퀴 세계여행', '그림 웹툰', '시사 이슈',
            'IT 트렌드', '사진 촬영', '취향저격 영화리뷰', '뮤직 인사이드',
            '육아 이야기', '요리 레시피', '건강 운동', '멘탈관리 심리탐구',
            '문화 예술', '건축 설계', '인문학 철학','쉽게 읽는 역사',
            '우리집 반려동물' , '오늘은 이런 책', '직장인 현실 조언', '디자인 스토리',
            '감성 에세이']
        st.write("")
        status = st.radio("분류가 알맞게 되었는지 알려주세요!", ("<select>","맞춤", "틀림")) # <select> 기본값

        if status == "맞춤" : # 정답일 경우
            st.write("분류가 알맞게 되었군요! 추천시스템을 이용해보세요")
            label,proba_max = classify(document)

            ## 해당 글에 해당하는 keyword리스트를 가진 dictionary load
            cur_dir = os.path.dirname(__file__)
            category_dict = pickle.load(open(os.path.join(cur_dir,'pkl_objects','keyword_dict_renew.txt'), 'rb'))
            ## 추천 시스템 부분 시작
            st.write('---')
            st.write("## 추천 시스템")
            st.write("선택하신 키워드를 기반으로 다른 작가분의 글을 추천해드려요.")
            select_category = st.multiselect("keyword를 선택하세요.",get_categories(label,category_dict))
            st.write(len(select_category), "가지 keyword를 선택했습니다.")

            keyword_submit_button = st.button("keyword 선택 완료",key='select_category') # submit 버튼

            if keyword_submit_button: ## keyword 선택 완료 시
                df = load_data()
                keyword_count_vect = load_keyword_count_vect()
                keyword_mat = load_keyword_mat()


                st.write("")
                st.write("")
                st.write("키워드 트렌드")
                line_chart_df = keyword_trend_chart(df,select_category)
                st.line_chart(line_chart_df)

                select_category_joined = (' ').join(select_category)

                recommended_text = find_sim_document(df, keyword_count_vect, keyword_mat, select_category_joined, top_n=5)

                st.write("")
                st.write("<추천글 목록>")
                st.table(recommended_text)

                answer = 1 # 맞춤/틀림 여부
                sqlite_main(document, answer, label, None, select_category_joined) ## 결과 db 저장

        elif status == "틀림": # 오답일 경우
            st.write("분류가 잘못되었군요. 피드백을 주신다면 다음부턴 틀리지 않을거예요.")
            label,proba_max = classify(document)
            category_correction = st.selectbox("category 수정하기", category_list) # 오답일 경우 정답을 새로 입력받음
            if category_correction != "<select>": # 오답 수정 부분이 입력 받았을 경우 (default가 아닐경우 => 값을 입력받은 경우)
                st.write("피드백을 주셔서 감사합니다. 추천 시스템을 이용해보세요")

                ## 해당 글에 해당하는(수정 된 정답라벨) keyword리스트를 가진 dictionary load
                cur_dir = os.path.dirname(__file__)
                category_dict = pickle.load(open(os.path.join(cur_dir,'pkl_objects','keyword_dict_renew.txt'), 'rb'))

                st.write('---')
                st.write("## 추천 시스템")
                st.write("선택하신 키워드를 기반으로 다른 작가분의 글을 추천해드려요.")
                select_category = st.multiselect("keyword를 선택하세요.",get_categories(category_correction,category_dict))
                st.write(len(select_category), "가지 keyword를 선택했습니다.")

                keyword_submit_button = st.button("keyword 선택 완료",key='select_category') # submit 버튼

                if keyword_submit_button: ## keyword 선택 완료 시
                    df = load_data()
                    keyword_count_vect = load_keyword_count_vect()
                    keyword_mat = load_keyword_mat()

                    st.write("")
                    st.write("")
                    st.write("키워드 트렌드")
                    line_chart_df = keyword_trend_chart(df,select_category)
                    st.line_chart(line_chart_df)

                    select_category_joined = (' ').join(select_category)

                    recommended_text = find_sim_document(df, keyword_count_vect, keyword_mat, select_category_joined, top_n=5)

                    st.write("")
                    st.write("<추천글 목록>")
                    st.table(recommended_text)

                    answer = 0 # 맞춤/틀림 여부
                    sqlite_main(document, answer, label, category_correction, select_category_joined) ## 결과 db 저장

    ## code review 페이지.
    elif app_mode == "전체 Code":
        st.write("show code")





if __name__ == "__main__":
    main()
