파일 구조:
- SNU_challenge
  - Code
    - utils
      - common
        - loss_function.py
        - utils.py
      - data
        - load_data.py
        - transforms.py
      - learning
        - train_part.py
        - test_part.py
      - model
        - hinet.py
        - unet.py
    - train.py
    - evaluate.py
    - leaderboard_eval.py
  - Data
    - train
    - val
    - image_Leaderboard


데이터 준비:
Data/train에 brain1.h5 ~ brain407.h5를 넣는다.  
Data/val에 리더보드 데이터인 brain_test1.h5 ~ brain_test50.h5를 넣는다.
Data/image_Leaderboard에 최종평가 데이터를 넣는다.

코드 실행:
'''bash
cd Code
python3 train.py
python3 evaluate.py
python3 leaderboard_eval.py
'''

추가 사항:

train.py는 약 15시간이 소요됨 (GPU마다 상이할 수 있음)
다시 훈련하고 평가하기 위해서는, 새로 생성된 result 파일을 삭제하고 위의 코드 실행을 다시 하면 된다.
train.py를 실행하는 중, 모든 로스가 1로 뜨면서 마치 훈련이 제대로 되고 있지 않은 것처럼 보이는 현상이 발생할 수 있다. 이 경우 훈련을 중단하지 말고 그냥 두면 된다.