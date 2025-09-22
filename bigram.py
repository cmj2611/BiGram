import sys
import argparse
import os
import re
import pickle
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import random

class BigramModel:
    def __init__(self):
        """
        Bi-gram 모델 초기화
        """
        self.bigrams = defaultdict(Counter)

    def preprocess_text(self, text: str) -> List[str]:
        """텍스트 전처리"""
        # 문장 부호와 특수문자 정리
        text = re.sub(r'[^\w\s가-힣]', '', text)
        # 여러 공백을 하나로 통일
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # 토큰화 (공백 기준)
        tokens = text.split()
        return tokens

    def get_bigrams(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """토큰 리스트에서 bi-gram 생성"""
        bigrams = []
        # 시작 토큰 추가
        padded_tokens = ['<START>'] + tokens + ['<END>']

        for i in range(len(padded_tokens) - 1):
            bigram = (padded_tokens[i], padded_tokens[i + 1])
            bigrams.append(bigram)

        return bigrams

    def train(self, text_file: str, max_lines: Optional[int] = None):
        """텍스트 파일로부터 bi-gram 모델 학습"""
        print("Bi-gram 모델 학습 시작...")

        if not os.path.exists(text_file):
            raise FileNotFoundError(f"학습 데이터 파일을 찾을 수 없습니다: {text_file}")

        line_count = 0
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if max_lines and line_count >= max_lines:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    tokens = self.preprocess_text(line)
                    if len(tokens) < 2:  # 너무 짧은 문장은 제외
                        continue

                    # bi-gram 생성 및 카운팅
                    bigrams = self.get_bigrams(tokens)
                    for bigram in bigrams:
                        first_word = bigram[0]
                        second_word = bigram[1]
                        self.bigrams[first_word][second_word] += 1

                    line_count += 1
                    if line_count % 10000 == 0:
                        print(f"처리된 라인: {line_count}")

        except UnicodeDecodeError:
            raise ValueError(f"파일 인코딩 오류: {text_file} (UTF-8로 저장되어야 합니다)")

        print(f"학습 완료. 총 {line_count}줄 처리")
        print(f"Bi-gram 패턴 수: {len(self.bigrams)}")

        if line_count == 0:
            raise ValueError("유효한 학습 데이터가 없습니다.")

    def get_next_word_probabilities(self, word: str) -> Dict[str, float]:
        """주어진 단어에 대해 다음 단어의 확률 분포 계산"""
        if word not in self.bigrams:
            return {}

        word_counts = self.bigrams[word]
        total_count = sum(word_counts.values())

        probabilities = {}
        for next_word, count in word_counts.items():
            probabilities[next_word] = count / total_count

        return probabilities

    def complete_sentence(self, partial_sentence: str, max_words: int = 20, temperature: float = 1.0) -> str:
        """불완전한 문장을 완성"""
        if not partial_sentence.strip():
            return ""

        tokens = self.preprocess_text(partial_sentence)
        if not tokens:
            return partial_sentence

        generated_tokens = tokens[:]

        for _ in range(max_words):
            # 마지막 단어를 context로 사용
            last_word = generated_tokens[-1]

            # 다음 단어 예측
            probabilities = self.get_next_word_probabilities(last_word)

            if not probabilities:
                break

            # <END> 토큰이 가장 확률이 높으면 종료
            if '<END>' in probabilities and probabilities['<END>'] > 0.3:
                break

            # Temperature를 적용한 샘플링
            if temperature <= 0:
                # 가장 확률이 높은 단어 선택 (<END> 제외)
                filtered_probs = {k: v for k, v in probabilities.items() if k != '<END>'}
                if not filtered_probs:
                    break
                next_word = max(filtered_probs.items(), key=lambda x: x[1])[0]
            else:
                # Temperature를 적용한 확률적 샘플링
                words = [w for w in probabilities.keys() if w != '<END>']
                if not words:
                    break

                probs = [probabilities[w] for w in words]

                # Temperature 적용
                if temperature != 1.0:
                    probs = [p ** (1.0 / temperature) for p in probs]
                    total = sum(probs)
                    if total == 0:
                        break
                    probs = [p / total for p in probs]

                # 확률적 선택
                try:
                    next_word = random.choices(words, weights=probs)[0]
                except IndexError:
                    break

            generated_tokens.append(next_word)

        return ' '.join(generated_tokens)

    def save_model(self, filepath: str):
        """모델을 파일로 저장"""
        model_data = { 'bigrams': dict(self.bigrams) }

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            print(f"모델이 {filepath}에 저장되었습니다.")
        except Exception as e:
            raise IOError(f"모델 저장 실패: {e}")

    def load_model(self, filepath: str):
        """파일에서 모델 로드"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {filepath}")

        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.bigrams = defaultdict(Counter, model_data['bigrams'])

        except Exception as e:
            raise IOError(f"모델 로드 실패: {e}")

def train_command(args):
    """학습 명령 실행"""
    print("=== Bi-gram 모델 학습 ===")

    model = BigramModel()

    try:
        model.train(args.input_file, max_lines=args.max_lines)
        model.save_model(args.output_model)
        print("학습이 완료되었습니다!")

    except Exception as e:
        print(f"학습 중 오류 발생: {e}", file=sys.stderr)
        return 1

    return 0

def batch_command(args):
    """배치 문장 완성 명령 실행"""
    try:
        model = BigramModel()
        model.load_model(args.model_file)

    except Exception as e:
        print(f"모델 로드 실패: {e}", file=sys.stderr)
        return 1

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]

    except Exception as e:
        print(f"입력 파일 읽기 실패: {e}", file=sys.stderr)
        return 1

    results = []

    print(f"총 {len(sentences)}개 문장 처리 중...")

    for i, sentence in enumerate(sentences, 1):
        try:
            completed = model.complete_sentence(
                sentence,
                max_words=args.max_words,
                temperature=args.temperature
            )
            results.append(f"입력: {sentence}")
            results.append(f"완성: {completed}")
            results.append("")

            if i % 100 == 0:
                print(f"진행률: {i}/{len(sentences)}")

        except Exception as e:
            results.append(f"입력: {sentence}")
            results.append(f"오류: {e}")
            results.append("")

    # 결과 저장
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(results))
        print(f"결과가 {args.output_file}에 저장되었습니다.")

    except Exception as e:
        print(f"결과 저장 실패: {e}", file=sys.stderr)
        return 1

    return 0

def main():
    parser = argparse.ArgumentParser(
        description="Bi-gram 기반 텍스트 완성 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 모델 학습
  python ngram.py train -i train_data.txt -o model.pkl

  # 배치 문장 완성
  python ngram.py batch -m model.pkl -i input.txt -o output.txt
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='사용할 명령')

    # 학습 명령
    train_parser = subparsers.add_parser('train', help='Bi-gram 모델 학습')
    train_parser.add_argument('-i', '--input-file', required=True,
                            help='학습할 텍스트 파일')
    train_parser.add_argument('-o', '--output-model', required=True,
                            help='저장할 모델 파일명')
    train_parser.add_argument('--max-lines', type=int,
                            help='최대 학습 라인 수 (전체 사용시 생략)')

    # 배치 완성 명령
    batch_parser = subparsers.add_parser('batch', help='배치 문장 완성')
    batch_parser.add_argument('-m', '--model-file', required=True,
                            help='로드할 모델 파일')
    batch_parser.add_argument('-i', '--input-file', required=True,
                            help='완성할 문장들이 있는 입력 파일')
    batch_parser.add_argument('-o', '--output-file', required=True,
                            help='결과를 저장할 출력 파일')
    batch_parser.add_argument('--max-words', type=int, default=15,
                            help='최대 생성 단어 수 (기본값: 15)')
    batch_parser.add_argument('--temperature', type=float, default=0.7,
                            help='Temperature 값 (기본값: 0.7)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # 명령 실행
    if args.command == 'train':
        return train_command(args)
    elif args.command == 'batch':
        return batch_command(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
