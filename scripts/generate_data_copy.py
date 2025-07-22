# scripts/generate_data.py
"""
python scripts/generate_data.py
"""

import sys
import os

# --- このブロックを追加 ---
# このスクリプト(generate_data.py)の場所を基準に、
# 親ディレクトリ(block_stacking_ml)に上がり、そこから'src'フォルダへのパスを作成
SRC_DIR = os.path.join(os.path.dirname(__file__), '..', 'src')
# Pythonのモジュール検索パスのリストに、作成したパスを追加
sys.path.append(SRC_DIR)
# --- ここまで ---

# これで、Pythonは 'src' フォルダの中を直接探しに行けるようになります
from environment_copy import DataCollectorGUI

def main():
    """
    データ収集GUIを初期化して実行するメイン関数。
    """
    # Dockerのボリュームマウント設定により、このパスはホストの data/01_raw/ に対応する
    save_path = "data/01_raw/stacking_data.csv"
    
    # GUIツールのインスタンスを作成
    app = DataCollectorGUI(save_path)
    
    # アプリケーション（ゲーム）を実行
    print("アプリケーションを実行します...")
    app.run()
    print("アプリケーションが終了しました。")

if __name__ == '__main__':
    # このスクリプトが直接実行された場合にmain関数を呼び出す
    main()