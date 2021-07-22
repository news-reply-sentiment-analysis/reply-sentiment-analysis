# koelectra를 이용한 댓글 감정 분석

## 1. 모델 설명

koelectra의 small-v2버전을 사용했으며, 데이터는 네이버 영화 댓글을 사용함.  
학습 한 후 정확도는 약 86%정도였다.

## 2. 모델 실행 방법

현재 학습된 모델이 model.pt파일로 만들어져 있는 상태이다.  
이 모델을 `nsmc_test.py`에서 사용하게 된다.

~~~shell
> python nsmc_test.py
~~~

실행하게 되면 처음에 모델을 불러오는데 시간이 좀 걸린다.(GPU로 하면 빠르겠지만, 일단 cpu로 구동되는 상태)  
실행되면 여러 영어가 _주~욱_ 나오게 되고 아래와 같은 문구가 나오면 내가 원하는 문장을 입력한다. (한글로)

> please input reply that know sentiment >


제일 마지막의 `tensor([0])` 혹은 `tensor([1])`로 나오게 되는데 이를 보면 감정을 알 수 있다.
(0은 부정, 1은 긍정)
그 밑에 이를 부정은 **Negative**, 긍정은 **Positive**라고 출력한다.

현재 테스트 코드에서 댓글 입력은 무한 루프이며, quit라는 문구를 입력하면 종료된다.

## 기타 내용

- cuda version == 10.1
- python version == 3.8.0
