# < Brunch Networking >

<br> 
<br>

<b> - 브런치 작가를 위한 Text 분류모델과 추천시스템 - </b>

글쓰기 플랫폼 "Brunch"를 아시나요?
브런치는 "누구나 작가가 될 수 있다"를 모토로 하는 콘텐츠 퍼블리싱 플랫폼입니다.브런치가 주목받는 이유 중 하나는 전문 작가가 아닌 일반인들도 자신들의 글을 연재할 수 있는 서비스를 제공해준다는것인데요, 이러한 서비스를 바탕으로 현재 브런치에는 수 많은 일반인 작가들이 다양한 글을 게시하고 독자들을 끌어들이고 있습니다. <br><br>
제가 만든 "Brunch Networking"의 목적은 "머신러닝"을 활용하여 작가들에게 편리함을 제공하는것 입니다. <br>
"브런치 네트워킹"은 데이터 수집부터 전처리, 모델링, API를 활용한 실제 서비스까지 머신러닝 프로젝트의 전 과정을 다루고 있습니다. 

<br>

## 최종 결과물 실행 영상(이미지 클릭)

먼저, 최종 결과물을 실행 영상으로 확인해보겠습니다.

[![Video Label](https://img.youtube.com/vi/RpEBgY3_stA/0.jpg)](https://youtu.be/RpEBgY3_stA)

<br>

## < 전체 흐름도 >

<br>

프로젝트의 전체적인 흐름입니다. BeautifulSoup과 Selenium을 사용하여 브런치의 게시글 20만개를 수집하였고, 한글 형태소 분석기인 Konlpy를 활용하여 수집된 텍스트 데이터를 전처리하였습니다. 이후에 여러가지 분류모델을 실험하고 가장 적합한 모델을 최종 선택하여 Streamlit을 활용해 API형태로 구현하였습니다.

![메인흐름](https://user-images.githubusercontent.com/35517797/81902112-8f7e4080-95fa-11ea-8954-1ab9952ec4e6.PNG)

<br>

## < 모델 학습 및 선택 >

<br>

모델 학습 과정과 최종 모델 선택에 관한 부분입니다. "브런치 네트워킹"의 모델이 풀어야할 과제는 18개의 클래스를 가지는 다중분류 문제입니다. 

* 현재 브런치의 카테고리는 총 24개입니다. 이 중에서 디자인, 그림웹툰 등 텍스트 보다는 이미지 위주의 글이 대다수인 6개의 카테고리를 제외한 18개의 카테고리만을 사용하였습니다. 

![모델학습 및 선택](https://user-images.githubusercontent.com/35517797/81902305-e08e3480-95fa-11ea-88bb-b151e2a45848.PNG)

## < 분류모델 API >
![분류모델](https://user-images.githubusercontent.com/35517797/81902312-e4ba5200-95fa-11ea-82ea-8109261abbfa.PNG)

## < 키워드 기반 text 추천 >
![추천1](https://user-images.githubusercontent.com/35517797/81902318-eab03300-95fa-11ea-9b23-8061e83324c7.PNG)

## < 문서 유사도 기반 text 추천 > 
![추천2](https://user-images.githubusercontent.com/35517797/81902325-ec79f680-95fa-11ea-9f97-5c5c35322ab4.PNG)
