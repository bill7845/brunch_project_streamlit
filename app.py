import streamlit as st
import pandas as pd
import numpy as np

import pickle
import sqlite3
import os
from konlpy.tag import Twitter


def load_clf():
    cur_dir = os.path.dirname(__file__)
    clf = pickle.load(open(os.path.join(cur_dir,'pkl_objects','classifier.pkl'), 'rb'))
    return clf

twitter = Twitter()
def tw_tokenizer(text):
    tokens_ko = twitter.morphs(text)
    return tokens_ko

def load_tfidf_matrix():
    cur_dir = os.path.dirname(__file__)
    tfidf_matrix_train = pickle.load(open(os.path.join(cur_dir,'pkl_objects','tfidf_matrix_train.pkl'), 'rb'))
    return tfidf_matrix_train

def classify(document):
    label = {0:'지구한바퀴_세계여행', 1:'그림·웹툰', 2:'시사·이슈', 3:'IT_트렌드',4:'사진·촬영', 5:'취향저격_영화_리뷰', 6:'뮤직_인사이드', 7:'육아_이야기', 8:'요리·레시피',
    9:'건강·운동', 10:'멘탈_관리_심리_탐구', 11:'문화·예술', 12:'건축·설계',
    13:'인문학·철학',14:'쉽게_읽는_역사', 15:'우리집_반려동물', 16:'오늘은_이런_책',
    17:'직장인_현실_조언', 18:'디자인_스토리',19:'감성_에세이'}

    tfidf_matrix_train = load_tfidf_matrix()
    clf = load_clf()

    X = tfidf_matrix_train.transform([document])
    y = clf.predict(X)[0]

    proba = clf.predict_proba(X)
    proba_max = np.max(proba)

    return label[y],proba_max

def main():
    st.title("Wellcome!")

    document = st.text_area("text를 입력하세요!")
    if st.button("Submit", key='document'):
        # result = document.title()
        label,proba_max = classify(document)
        st.write('이 글은 %f의 확률로 %s 입니다' %(proba_max,label))

    status = st.radio("맞춤 / 틀림", ("<select>","correct", "incorrect"))

    category_correction = st.empty()
    category_correction = " OPTIMIZE: Category! "

    category_list = ['<select>','지구한바퀴_세계여행', '그림·웹툰', '시사·이슈',
        'IT_트렌드', '사진·촬영', '취향저격_영화_리뷰', '뮤직_인사이드',
        '육아_이야기', '요리·레시피', '건강·운동', '멘탈_관리_심리_탐구',
        '문화·예술', '건축·설계', '인문학·철학','쉽게_읽는_역사',
        '우리집_반려동물' , '오늘은_이런_책', '직장인_현실_조언', '디자인_스토리',
        '감성_에세이']
    if status == "incorrect" :
        category_correction = st.selectbox("category 수정하기", category_list)
        if category_correction != "<select>":
            st.write("You selected this option ",category_correction)
    elif status == "correct":
        st.write("correct!")
        category_correction = label

    # choose_category = st.selectbox("",["NONE","Decision Tree", "Neural Network", "K-Nearest Neighbours"])

	# if st.checkbox('Show Raw Data'):
	# 	st.subheader("Showing raw data---->>>")
	# 	st.write('hello')



if __name__ == "__main__":
    main()
