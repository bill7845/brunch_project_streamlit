## <b> 1. 데이터 수집 </b>

<br>

Bruch Networking 프로젝트를 진행하며 경험한 데이터 수집-정제-분석-적용 전체 과정을 설명하고자 합니다. 전체 code는 github에서 확인하실 수 있습니다.

<br>
<br>


<img src = "https://user-images.githubusercontent.com/35517797/80509879-f1149d00-89b4-11ea-939d-5eb1af734319.png" height="300" width="650px">

<br>
<br>
<br>

먼저, 프로젝트를 위해 필요한 데이터를 정의합니다. text 자동 분류와 추천시스템 구현이 주된 목적이므로 아래와 같이 필요한 데이터를 정의했습니다.

<br>

* <b> 전체 20개 카테고리의 각 카테고리별 게시글 </b>
* <b> 게시글 별 정보(제목,발행일,공유횟수,좋아요수,keyword,댓글 ...) </b>

<br>
<br>

이렇게 필요 데이터를 정의한 후 본격적으로 크롤링 코드를 구현합니다. 브런치의 카테고리별 게시글page의 경우 "무한스크롤" javaScript가 구현되어 있습니다.
javaScript를 제어하기 위해서는 BeautifulSoup만으로는 불가능하므로 시간이 더 소요될 수 있지만 Selenium패키지를 사용해야 합니다.

<br>

<b> 1. 카테고리별 게시글 목록 page => 게시글 별 작가 id(url) parsing </b> <br>
<b> 2. 각 게시글 page => text, 제목, 발행일 등의 정보를 parsing </b>

<br>
<br>

위 두 과정을 코드와 함께보면 다음과 같습니다. 첫번째로, 개별 카테고리 -> 전체 url 정보를 수집하여 pickle 형식으로 저장합니다.

~~~python
###################################################################
########################### 크롤링 1단계 ###########################
###################################################################
import time
import requests
import pickle
from bs4 import BeautifulSoup

# 카테고리 별 게시글 리스트 페이지에서 유저별 id 파싱 (*전체 24개 카테고리)
# 각 페이지별 무한 스크롤 javaScript 제어를 위한 셀레니움 기능 사용
def get_user_list(base_url):

    # 크롬드라이버 설정
    chromedriver = 'C:/selenium/chromedriver.exe' # chromedriver path
    driver = webdriver.Chrome(chromedriver)
    driver.get(base_url) # url acess

    SCROLL_PAUSE_TIME = 10 # 무한스크롤 멈춤 현상 예방
    last_height = driver.execute_script("return document.body.scrollHeight")
    i = 0
    while True:
        # Scroll down to bottom                                                      
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)                                                
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight-50);")  
        time.sleep(SCROLL_PAUSE_TIME)

        # Calculate new scroll height and compare with last scroll height            
        new_height = driver.execute_script("return document.body.scrollHeight")
        i+=1

        if i == 350: # 각 카테고리별 스크롤 횟수를 350번으로 제어                                                
            break
        last_height = new_height

    # 각 페이별 350번 스크롤 후, BeautifulSoup으로 "전체 게시글 url parsing"
    html = driver.page_source
    soup = BeautifulSoup(html,'html.parser')
    a_tags = soup.select('#wrapArticle > div.wrap_article_list.\#keyword_related_contents > ul > li > a.link_post')

    save_href = []
    for a_tag in a_tags :
        save_href.append(a_tag['href'])

    driver.close()

    return save_href # 개별 카테고리별 전체 게시글 url 반환

category_list = ['지구한바퀴_세계여행?q=g','그림·웹툰?q=g','시사·이슈?q=g','IT_트렌드?q=g','사진·촬영?q=g','',
                 '취향저격_영화_리뷰?q=g','오늘은_이런_책?q=g','뮤직_인사이드?q=g','글쓰기_코치?q=g','직장인_현실_조언?q=g'
                 ,'스타트업_경험담?q=g','육아_이야기?q=g','요리·레시피?q=g','건강·운동?q=g','멘탈_관리_심리_탐구?q=g',
                 '디자인_스토리?q=g','문화·예술?q=g','건축·설계?q=g','인문학·철학?q=g','쉽게_읽는_역사?q=g','우리집_반려동물?q=g',
                 '멋진_캘리그래피?q=g','사랑·이별?q=g','감성_에세이?q=g']


# 전체 24개 카테고리별로 돌며 게시글 url 파싱하여 pickle로 저장
for category in category_list:
    each_user_id = []
    each_user_id = get_user_list("https://brunch.co.kr/keyword/"+category)
    with open(category[:-4]+'_userId.txt','wb') as f:
        pickle.dump(each_user_id,f)
~~~

<br>

이미지이미지...
크롤링 두번째 단계로, 저장한 게시글 url pickle파일을 불러와 각각의 게시글 속 정보를 수집합니다.

~~~python
###################################################################
########################### 크롤링 2단계 ###########################
###################################################################
## url pickle load
pickles = ['지구한바퀴_세계여행?q=g','그림·웹툰?q=g','시사·이슈?q=g','IT_트렌드?q=g','사진·촬영?q=g',
                 '취향저격_영화_리뷰?q=g','오늘은_이런_책?q=g','뮤직_인사이드?q=g','글쓰기_코치?q=g','직장인_현실_조언?q=g'
                 ,'스타트업_경험담?q=g','육아_이야기?q=g','요리·레시피?q=g','건강·운동?q=g','멘탈_관리_심리_탐구?q=g',
                 '디자인_스토리?q=g','문화·예술?q=g','건축·설계?q=g','인문학·철학?q=g','쉽게_읽는_역사?q=g','우리집_반려동물?q=g',
                 '멋진_캘리그래피?q=g','사랑·이별?q=g','감성_에세이?q=g']

writer_list = []
for file in pickles:
#     print(file)
    with open('C:/Users/KIHyuk/Desktop/brunch_data/user_id/'+file[:-4]+"_userId.txt","rb") as fr:
        writers = pickle.load(fr)
    writer_list.append(writers)  ## [[카테고리1 게시글 url...],[카테고리2 게시글 url], ....[카테고리24 게시글 url]]

## 게시글 속 정보 수집
def def_craw(writer):
    json_data = {}
    data = []
    res_text = []
    tag_keyword=[]

    tag_title,tag_nickname,tag_publish_date,tag_url,tag_url_plink = None,None,None,None,None
    tag_share,tag_like = str,str
    for url in writer:
        if res_text == []: # 첫 시작에러 방지
            pass
        else :
            # to json
            json_data['title'] = tag_title  
            json_data['nickname'] = tag_nickname
            json_data['publish_date'] = tag_publish_date
            json_data['keyword'] = tmp_keyword   
            json_data['like'] = tag_like # like 없는 경우 ''
            json_data['share'] = tag_share # share 없는 경우 None
            json_data['comment'] = tag_comment # comment 없는 경우 ''
            json_data['url'] = tag_url
            json_data['url_plink'] = tag_url_plink
            json_data['text'] = res_text

        data.append(json_data)

        json_data = {} # 누적방지 초기화
        tmp_keyword = [] # 누적방지 초기화
        res_text = [] # 누적방지 초기화

        # beautifulsoup
        html = requests.get('https://brunch.co.kr{text_url}'.format(text_url=url))
        soup = BeautifulSoup(html.text, 'html.parser')

        if soup.find('title').text == "brunch":
            pass
        else:
            tag_title = soup.find('title').text # 게시글 title
            tag_url = soup.find("meta",property='og:url')['content'] # 게시글 본주소
            tag_nickname = soup.find("meta",{'name':'article:media_name'})['content'] # 작가 nickname
            tag_url_plink = soup.find("meta",property='dg:plink')['content'] # 암호주소? # 모바일?
            tag_publish_date = soup.find("meta",property='article:published_time')['content'] # 발행일
            tag_keyword = soup.find_all('a',href=re.compile('/keyword')) # 게시글 키워드
            tag_like = soup.find('span',{'class':'f_l text_like_count text_default text_with_img_ico ico_likeit_like #like'}) #좋아요 수
            tag_share = soup.find('span',{'class':'f_l text_share_count text_default text_with_img_ico'}) # 공유 수
            tag_comment = soup.find('span',{'class':'f_l text_comment_count text_default text_with_img_ico'}) # 댓글 수
            text_h4 = soup.find_all(class_='wrap_item item_type_text')

            for text in text_h4:
                res_text.append(text.text)

            if tag_like == None:
                tag_like = "0"
            else:
                tag_like = tag_like.text # 좋아요 수

            if tag_share == None:
                tag_share == "0"
            else:
                tag_share = tag_share.text # 공유 수

            if tag_comment == None:
                tag_comment =="0"
            else:
                tag_comment = tag_comment.text

            for keyword in tag_keyword:
                tmp_keyword.append(keyword.text)

    return data ## 수집한 정보를 담은 dictionary로 반환

# categories = ['지구한바퀴_세계여행','그림·웹툰','시사·이슈','IT_트렌드','사진·촬영',
#                  '취향저격_영화_리뷰','오늘은_이런_책','뮤직_인사이드','글쓰기_코치','직장인_현실_조언'
#                  ,'스타트업_경험담','육아_이야기','요리·레시피','건강·운동','멘탈_관리_심리_탐구',
#                  '디자인_스토리','문화·예술','건축·설계','인문학·철학','쉽게_읽는_역사','우리집_반려동물',
#                  '멋진_캘리그래피','사랑·이별','감성_에세이']


## 카테고리 -> 게시글의 순서로 2차 크롤링 진행
## 카테고리별로 정보를 담은 json 형식으로 저장(총 24개 json file)
for idx,writer in enumerate(writer_list):
    to_json = None
    data = def_craw(writer) # 2단계 크롤링 실행

    del data[0]
    del data[0]

    to_json = OrderedDict()
    to_json['name'] = categories[idx] # category name
    to_json['version'] = "2020-04-21"
    to_json['data'] = data

    with open(categories[idx]+".json","w") as make_file:
        json.dump(to_json,make_file)
~~~

이미지이미지..


카테고리별로 게시글의 수가 다르기 때문에 무한스크롤의 횟수를 조건으로 각각 어느정도의 데이터를 수집하였습니다. 수집한 데이터를 간단한 전처리를 거쳐 DataFrame형식으로
만들었고 아래의 이미지와 같습니다. 전처리 부분은 바로 이어서 설명하겠습니다.

<img src = "https://user-images.githubusercontent.com/35517797/80506298-587c1e00-89b0-11ea-9602-c63c1526b109.PNG" height="330" width="700px">

<br>
<br>

## 2. Text 데이터 전처리

<br>

크롤링 첫 단계에서 수집된 데이터는 여러가지 "잔"처리를 해주어야 합니다. 제가 만든 크롤러의 경우에 2단계의 과정을 거쳐야 하기 때문에 pickle과 json형식을 주로 사용하였으며, pandas 라이브러리를 주로 활용하였습니다. 이제 기초적인 "잔"처리 단계가 끝나면 위 이미지와 같은 데이터 형태를 얻게되고, text분석을 위한 전처리 단계에 돌입해야 합니다.

### 2.1 토큰화(Tokenization)

토큰화란, 분석하고자하는 문서를 문장 혹은 단어 단위로 분리해내는것을 의미합니다.
