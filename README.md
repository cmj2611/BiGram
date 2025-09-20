# Bigram 텍스트 완성 프로그램

Bigram 모델을 사용하여 불완전한 문장을 자동으로 완성하는 프로그램입니다.

## 사용법

### 1. 모델 학습

텍스트 데이터로부터 Bigram 모델을 학습합니다.

```bash
python bigram.py train -i <학습_데이터_파일> -o <모델_파일>
```

**옵션:**
- `-i, --input-file`: 학습할 텍스트 파일 (필수)
- `-o, --output-model`: 저장할 모델 파일명 (필수)
- `--max-lines`: 최대 학습 라인 수 (선택사항, 생략시 전체 사용)

**예시:**
```bash
python bigram.py train -i train_data.txt -o model.pkl
python bigram.py train -i train_data.txt -o model.pkl --max-lines 10000
```

### 2. 배치 문장 완성

학습된 모델을 사용하여 여러 문장을 일괄 완성합니다.

```bash
python bigram.py batch -m <모델_파일> -i <입력_파일> -o <출력_파일>
```

**옵션:**
- `-m, --model-file`: 로드할 모델 파일 (필수)
- `-i, --input-file`: 완성할 문장들이 있는 입력 파일 (필수)
- `-o, --output-file`: 결과를 저장할 출력 파일 (필수)
- `--max-words`: 최대 생성 단어 수 (기본값: 15)
- `--temperature`: Temperature 값 (기본값: 0.7, 낮을수록 보수적)

**예시:**
```bash
python bigram.py batch -m model.pkl -i test_input.txt -o test_output.txt
python bigram.py batch -m model.pkl -i test_input.txt -o test_output.txt --max-words 10 --temperature 0.5
```

## 파일 형식

### 학습 데이터 파일
- UTF-8 인코딩의 텍스트 파일
- 한 줄에 하나의 문장
- 빈 줄은 자동으로 무시됨

### 입력 파일
- UTF-8 인코딩의 텍스트 파일
- 한 줄에 하나의 불완전한 문장
- 예: "나는 오늘 저녁에", "이 영화는 정말"

### 출력 파일
- 각 입력 문장에 대해 다음 형식으로 저장:
  ```
  입력: 나는 오늘 저녁에
  완성: 나는 오늘 저녁에 친구와 함께 식사를 했다
  
  입력: 이 영화는 정말
  완성: 이 영화는 정말 재미있고 감동적이었다
  ```

## Temperature 값 설정

- **0.0**: 가장 확률이 높은 단어만 선택 (가장 보수적)
- **0.5**: 상당히 보수적인 선택
- **1.0**: 원래 확률 분포 사용
- **1.5 이상**: 더 창의적이지만 일관성이 떨어질 수 있음

## 도움말

```bash
python bigram.py -h              # 전체 도움말
python bigram.py train -h        # 학습 명령 도움말  
python bigram.py batch -h        # 배치 완성 명령 도움말
```
