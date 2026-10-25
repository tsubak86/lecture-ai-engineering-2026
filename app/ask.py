"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
# from google.genai import errors
from google.genai import errors, types  # ★「, types」を付け足しました

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


#def ask(question: str) -> str:
# 1. ask関数に temperature を受け取る仕組みを追加
def ask(question: str, temp_value: float) -> str:
    client = genai.Client(api_key=get_api_key())

    # ★ GoogleのAI用に温度（temperature）の設定指示書を作ります
    config_setting = types.GenerateContentConfig(temperature=temp_value)

    # ★ config= で指示書を一緒に送信します
    response = client.models.generate_content(
        model=MODEL, contents=question, config=config_setting
        )
    
    usage = response.usage_metadata
    print(
        f"[トークン数] 入力: {usage.prompt_token_count}, "
        f"出力: {usage.candidates_token_count}"
    )

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    #parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)

    # 2. 引数を「順番」ではなく「名前（オプション引数）」で指定できるように変更
    parser.add_argument("-q", "--question", default=DEFAULT_QUESTION)

    # ★ 温度（temperature）をコマンドから受け取る設定を追加（デフォルトは 1.0 に設定）
    parser.add_argument("-t", "--temperature", type=float, default=1.0)

    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[temperature] {args.temperature}")  # ★現在の温度を画面に表示
    print(f"[質問] {args.question}\n")

    try:
        #print(ask(args.question))
        # 3. ask関数を呼ぶときに、コマンドから受け取った温度も一緒に渡す
        print(ask(args.question, args.temperature))

    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
