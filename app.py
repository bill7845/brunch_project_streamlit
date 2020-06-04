#######################################################################
## Library import
#######################################################################
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pickle
import json
import sqlite3
import os
from konlpy.tag import Mecab

import sklearn
import konlpy
import matplotlib as mat
import seaborn as sns
import plotly as plt

# import sys
# print(sys.version)

## load csv
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('C:/Users/KIHyuk/Documents/GitHub/project_brunchNetworking_streamlit/pkl_objects/all_df.csv',index_col='Unnamed: 0')
    return df

## load count_vect
@st.cache(allow_output_mutation=True)
def load_keyword_count_vect():
    cur_dir = os.path.dirname(__file__)
    keyword_count_vect = pickle.load(open(os.path.join(cur_dir,'pkl_objects','keyword_count_vect.pkl'), 'rb'))
    return keyword_count_vect

## load keyword_mat
@st.cache(allow_output_mutation=True)
def load_keyword_mat():
    cur_dir = os.path.dirname(__file__)
    keyword_mat = pickle.load(open(os.path.join(cur_dir,'pkl_objects','keyword_mat.pkl'), 'rb'))
    return keyword_mat

## load classifier
@st.cache
def load_clf():
    cur_dir = os.path.dirname(__file__)
    clf = pickle.load(open(os.path.join(cur_dir,'pkl_objects','classifier_logistic.pkl'), 'rb'))
    return clf

## laod tfidf vector
@st.cache(allow_output_mutation=True)
def load_tfidf_train_vect():
    cur_dir = os.path.dirname(__file__)
    tfidf_train_vect = pickle.load(open(os.path.join(cur_dir,'pkl_objects','tfidf_train_vect.pkl'), 'rb'))
    return tfidf_train_vect

## laod tfidf matrix
@st.cache
def load_tfidf_train_matrix():
    cur_dir = os.path.dirname(__file__)
    tfidf_train_matrix = pickle.load(open(os.path.join(cur_dir,'pkl_objects','tfidf_train_matrix.pkl'), 'rb'))
    return tfidf_train_matrix

# 입력받은 문서를 분류하여 분류결과,분류확률 반환
def classify(document,tfidf_train_vect):
    label = {0:'육아_이야기', 1:'요리_레시피', 2:'건강_운동', 3:'멘탈관리_심리탐구',4:'문화_예술', 5:'인문학_철학', 6:'쉽게_읽는_역사',
     7:'우리집_반려동물', 8:'사랑_이별',
    9:'감성_에세이', 10:'지구한바퀴_세계여행', 11:'시사_이슈', 12:'IT_트렌드',
    13:'취향저격_영화리뷰',14:'오늘은_이런책', 15:'뮤직_인사이드', 16:'직장인_현실조언',
    17:'스타트업_경험담'}
    mecab = Mecab(dicpath='C:\\mecab\\mecab-ko-dic')

    document = [i[0] for i in mecab.pos(document) if ( ((i[1]=="NNG") or (i[1]=="NNP") and (len(i[0])>1)) )]
    document = " ".join(document)

    # tfidf_train_vect = load_tfidf_train_vect()
    clf = load_clf() # 분류모델 load
    X = tfidf_train_vect.transform([document]) # train vector에 맞춰 입력text vectorize
    y = clf.predict(X)[0]

    proba = clf.predict_proba(X)
    proba_max = np.max(proba)

    return label[y],proba_max,y

## category에 해당하는 keyword 목록 반환
def get_categories(label,dict):
    return tuple(dict[label]) # multiselect box에서 사용위해 tuple로 반환

## 추천 시스템_1 작성 글 기반
def find_sim_document(df, input_document, y, top_n=3):
    mecab = Mecab(dicpath='C:\\mecab\\mecab-ko-dic')

    cur_dir = os.path.dirname(__file__)
    each_tfidf_vect = pickle.load(open(os.path.join(cur_dir,'pkl_objects/each_vect',str(y)+'tfidf_vect.pkl'), 'rb'))
    tfidf_matrix = pickle.load(open(os.path.join(cur_dir,'pkl_objects/each_matrix',str(y)+'tfidf_matrix.pkl'), 'rb'))

    input_document = [i[0] for i in mecab.pos(input_document) if ( ((i[1]=="NNG") or (i[1]=="NNP") and (len(i[0])>1)) )]
    input_document = " ".join(input_document)

    input_document_mat = each_tfidf_vect.transform([input_document])
    document_sim = cosine_similarity(input_document_mat, tfidf_matrix)

    document_sim_sorted_ind = document_sim.argsort()[:,::-1]

    top_n_sim = document_sim_sorted_ind[:1,:(top_n)]
    top_n_sim = top_n_sim.reshape(-1)

    res_df = df[df['class'] == y].iloc[top_n_sim][['title','text','keyword','url']]
    res_df['text'] = res_df['text'].apply(lambda x : x[:300])

    return res_df

## 추천 시스템_2 Keyword 기반
def find_sim_keyword(df, count_vect, keyword_mat, input_keywords, top_n=3):
  input_keywords_mat = count_vect.transform(pd.Series(input_keywords)) # 입력 받은 키워드를 count_vectorizer

  keyword_sim = cosine_similarity(input_keywords_mat, keyword_mat) # cosine_similarity 계산

  keyword_sim_sorted_ind = keyword_sim.argsort()[:,::-1]

  top_n_sim = keyword_sim_sorted_ind[:1,:(top_n)]
  top_n_sim = top_n_sim.reshape(-1)

  res_df = df.iloc[top_n_sim][['title','text','keyword','url']]

  test_list = []
  for text in res_df['text']:
      text = re.sub(r'[xa0]','',text)
      text = re.sub(r'[^0-9a-zA-Zㄱ-힗]',' ',text)
      text = text[:200]
      test_list.append(text)
  res_df['text'] = test_list

  return res_df

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

## read markdown.md
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text(encoding='UTF8')

#######################################################################
## main
#######################################################################
def main():
    st.sidebar.title("Menu")
    app_mode = st.sidebar.selectbox("",["Home", "App 실행"])

    tfidf_train_vect = load_tfidf_train_vect()
    df = load_data()

    ## 개요 페이지. (시작 페이지)
    if app_mode == "Home":
        st.markdown(readme_text,unsafe_allow_html=True)

        video_file = open('테스트비디오.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

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
            label,proba_max,y = classify(document,tfidf_train_vect) ## classify 함수에 의해 라벨,확률값
            st.write('작성하신 글은 %d퍼센트의 확률로 \'%s\' 카테고리로 분류됩니다.' %(round((proba_max)*100),label))

        #######################################################################
        ## 2. 분류 결과에 대한 맞춤,틀림 여부 입력받음
        ##      2.1 정답일 경우
        ##      2.2 오답일 경우
        #######################################################################
        category_list = ['<select>','지구한바퀴_세계여행', '시사_이슈',
            'IT_트렌드', '취향저격_영화리뷰', '뮤직_인사이드',
            '육아_이야기', '요리_레시피', '건강_운동', '멘탈관리_심리탐구',
            '문화_예술', '인문학_철학','쉽게_읽는_역사',
            '우리집_반려동물' , '오늘은_이런책', '직장인_현실조언','스타트업_경험담',
            '감성_에세이','사랑_이별']
        st.write("")
        status = st.radio("분류가 알맞게 되었는지 알려주세요!", ("<select>","맞춤", "틀림")) # <select> 기본값

        if status == "맞춤" : # 정답일 경우
            st.write("분류가 알맞게 되었군요! 추천시스템을 이용해보세요 작성하신 글을 기반으로 다른 작가분의 글을 추천해드려요")
            label,proba_max,y = classify(document,tfidf_train_vect)
            # df = load_data()

            recommended_text = find_sim_document(df,document,y,top_n=3)

            st.write("")
            st.write("<작성글 기반 추천글 목록>")
            st.table(recommended_text)

            ## 해당 글에 해당하는 keyword리스트를 가진 dictionary load
            cur_dir = os.path.dirname(__file__)
            category_dict = pickle.load(open(os.path.join(cur_dir,'pkl_objects','keyword_dict.txt'), 'rb'))
            ## 추천 시스템 부분 시작
            st.write('---')
            st.write("## 추천 시스템")
            st.write("선택하신 키워드를 기반으로 다른 작가분의 글을 추천해드려요.")
            select_category = st.multiselect("keyword를 선택하세요.",get_categories(label,category_dict))
            print('선택완료')
            st.write(len(select_category), "가지 keyword를 선택했습니다.")

            keyword_submit_button = st.button("keyword 선택 완료",key='select_category') # submit 버튼

            if keyword_submit_button: ## keyword 선택 완료 시
                # df = load_data()
                keyword_count_vect = load_keyword_count_vect()
                keyword_mat = load_keyword_mat()


                st.write("")
                st.write("")
                st.write("키워드 트렌드")
                line_chart_df = keyword_trend_chart(df,select_category)
                st.line_chart(line_chart_df)

                select_category_joined = (' ').join(select_category)

                recommended_keyword = find_sim_keyword(df, keyword_count_vect, keyword_mat, select_category_joined, top_n=5)

                st.write("")
                st.write("<추천글 목록>")
                st.table(recommended_keyword)

                answer = 1 # 맞춤/틀림 여부
                sqlite_main(document, answer, label, None, select_category_joined) ## 결과 db 저장

        elif status == "틀림": # 오답일 경우
            st.write("분류가 잘못되었군요. 피드백을 주신다면 다음부턴 틀리지 않을거예요.")
            label,proba_max,y = classify(document,tfidf_train_vect)
            category_correction = st.selectbox("category 수정하기", category_list) # 오답일 경우 정답을 새로 입력받음
            if category_correction != "<select>": # 오답 수정 부분이 입력 받았을 경우 (default가 아닐경우 => 값을 입력받은 경우)
                st.write("피드백을 주셔서 감사합니다. 추천 시스템을 이용해보세요")

                ## 해당 글에 해당하는(수정 된 정답라벨) keyword리스트를 가진 dictionary load
                cur_dir = os.path.dirname(__file__)
                category_dict = pickle.load(open(os.path.join(cur_dir,'pkl_objects','keyword_dict.txt'), 'rb'))

                st.write('---')
                st.write("## 추천 시스템")
                st.write("선택하신 키워드를 기반으로 다른 작가분의 글을 추천해드려요.")
                select_category = st.multiselect("keyword를 선택하세요.",get_categories(category_correction,category_dict))
                st.write(len(select_category), "가지 keyword를 선택했습니다.")

                keyword_submit_button = st.button("keyword 선택 완료",key='select_category') # submit 버튼

                if keyword_submit_button: ## keyword 선택 완료 시
                    # df = load_data()
                    keyword_count_vect = load_keyword_count_vect()
                    keyword_mat = load_keyword_mat()

                    st.write("")
                    st.write("")
                    st.write("키워드 트렌드")
                    line_chart_df = keyword_trend_chart(df,select_category)
                    st.line_chart(line_chart_df)

                    select_category_joined = (' ').join(select_category)

                    recommended_keyword = find_sim_keyword(df, keyword_count_vect, keyword_mat, select_category_joined, top_n=3)

                    st.write("")
                    st.write("<추천글 목록>")
                    st.table(recommended_keyword)

                    answer = 0 # 맞춤/틀림 여부
                    sqlite_main(document, answer, label, category_correction, select_category_joined) ## 결과 db 저장




if __name__ == "__main__":
    main()
