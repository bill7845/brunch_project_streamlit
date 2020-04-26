import streamlit as st
# import awesome_streamlit as ast
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pickle
import sqlite3
import os
from konlpy.tag import Twitter

## load csv
@st.cache
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
twitter = Twitter()
def tw_tokenizer(text):
    tokens_ko = twitter.morphs(text)
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


## main 함수
def main():
    st.title("환영합니다 작가님!")

    st.sidebar.title("둘러보기")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        st.write("sss")
        # st.code(get_file_content_as_string("app.py"))
    elif app_mode == "Run the app":
        readme_text.empty()
        # run_the_app()

    document = st.text_area("text를 입력해주세요") ## text 입력란

    submit_button = st.button("submit",key='document') # submit 버튼

    #######################################################################
    ## 1. 문서 입력 후 submit 버튼 클릭 시 분류 모델에 의해 분류라벨,확률값 출력
    #######################################################################
    if submit_button:
        label,proba_max = classify(document)
        st.write('작성하신 text는 %d퍼센트의 확률로 \'%s\' 카테고리로 분류됩니다.' %(round((proba_max)*100),label))

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

        select_category = st.multiselect("keyword를 선택하세요.",get_categories(label,category_dict))
        st.write(len(select_category), "가지 keyword를 선택했습니다.")

        keyword_submit_button = st.button("keyword 선택 완료",key='select_category') # submit 버튼

        if keyword_submit_button: ## keyword 선택 완료 시
            df = load_data()
            keyword_count_vect = load_keyword_count_vect()
            keyword_mat = load_keyword_mat()

            select_category = (' ').join(select_category)

            recommended_text = find_sim_document(df, keyword_count_vect, keyword_mat, select_category, top_n=5)
            st.table(recommended_text)


    elif status == "틀림": # 오답일 경우
        st.write("분류가 잘못되었군요. 피드백을 주시면 감사하겠습니다.")
        category_correction = st.selectbox("category 수정하기", category_list) # 오답일 경우 정답을 새로 입력받음
        if category_correction != "<select>": # 오답 수정 부분이 입력 받았을 경우 (default가 아닐경우 => 값을 입력받은 경우)
            st.write("피드백을 주셔서 감사합니다.",category_correction)

            ## 해당 글에 해당하는(수정 된 정답라벨) keyword리스트를 가진 dictionary load
            cur_dir = os.path.dirname(__file__)
            category_dict = pickle.load(open(os.path.join(cur_dir,'pkl_objects','keyword_dict_renew.txt'), 'rb'))

            st.write('---')
            st.write("## 추천 시스템")
            select_category = st.multiselect("keyword를 선택하세요.",get_categories(category_correction,category_dict))
            st.write(len(select_category), "가지 keyword를 선택했습니다.")

            keyword_submit_button = st.button("keyword 선택 완료",key='select_category') # submit 버튼

            if keyword_submit_button: ## keyword 선택 완료 시
                df = load_data()
                keyword_count_vect = load_keyword_count_vect()
                keyword_mat = load_keyword_mat()

                select_category = (' ').join(select_category)

                recommended_text = find_sim_document(df, keyword_count_vect, keyword_mat, select_category, top_n=5)
                st.table(recommended_text)



if __name__ == "__main__":
    main()
