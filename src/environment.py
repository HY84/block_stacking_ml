# src/environment.py

import pygame
import csv
import random
import os
import json

# --- 定数設定 ---
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1200
BLOCK_UNIT_SIZE = 15 # 細かいグリッド1マスの描画サイズ

class DataCollectorGUI:
    def __init__(self, save_path):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("データ収集ツール")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        # --- グリッドと監視ポイントの定義 ---
        self.grid_width = 45
        self.grid_height = 90

        # ▼▼▼ 土台のX座標オフセットを定義 ▼▼▼
        foundation_x_offset = 5 # 土台全体を右に5マスずらす

        # ▼▼▼ 高さの記録対象となる、山と谷のX座標を定義 ▼▼▼
        self.monitoring_points_x = []
        num_peaks = 6
        peak_spacing = 6
        for i in range(num_peaks):
            peak_x = i * peak_spacing + 2 + foundation_x_offset # 山の頂点のX座標
            self.monitoring_points_x.append(peak_x)
            if i < num_peaks -1:
                valley_x = i * peak_spacing + 5 + foundation_x_offset # 谷の底のX座標
                self.monitoring_points_x.append(valley_x)
        # これで、[7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37] の11点が監視対象になる

        self.load_block_definitions()
        self.save_path = save_path
        self.init_csv()
        self.reset_game()

    def load_block_definitions(self):
        config_path = "configs/blocks.json"
        try:
            with open(config_path, 'r') as f:
                self.block_definitions = json.load(f)
        except FileNotFoundError:
            print(f"エラー: 設定ファイルが見つかりません: {config_path}")
            self.block_definitions = {}

    def reset_game(self):
        # ▼▼▼ リセット時に12を保存するロジックを追加 ▼▼▼
        if hasattr(self, 'placed_blocks') and self.placed_blocks:
            # 現在の盤面状態を取得
            state_before = self.board_state.copy()
            # 配置済みのブロックがある場合のみ、リセット操作を記録
            # placed_x = 12 はリセット操作を示す特別な値
            # グリットの計算の都合上12を記録するために41を渡す
            self.save_data(state_before, self.current_block_type, 41, 2)
            print("データ保存: リセット操作を12として記録しました。")

        # ▼▼▼ board_stateを監視ポイントの数で初期化 ▼▼▼
        self.board_state = [0] * len(self.monitoring_points_x)
        self.placed_blocks = {}
        self.game_over = False
        self.block_appearance_count = {block_type: 0 for block_type in self.block_definitions.keys()}
        self.draw_initial_foundation()
        self.new_block()

    def draw_initial_foundation(self):
        foundation_color = (80, 80, 80)
        num_peaks = 6
        peak_spacing = 6
        # ▼▼▼ 土台のX座標オフセットを定義 ▼▼▼
        foundation_x_offset = 5 # __init__と同じ値を指定
        for i in range(num_peaks):
            start_x = i * peak_spacing + foundation_x_offset
            if start_x + 5 > self.grid_width: break
            for j in range(5): self.placed_blocks[(start_x + j, 0)] = foundation_color
            for j in range(3): self.placed_blocks[(start_x + 1 + j, 1)] = foundation_color
            self.placed_blocks[(start_x + 2, 2)] = foundation_color
        self._update_board_state()
    
    def new_block(self):
        if self.game_over or not self.block_definitions: return

        # ▼▼▼ 出現回数が上限に達していないブロックのリストを作成 ▼▼▼
        available_blocks = [
            block_type for block_type, count in self.block_appearance_count.items()
            if count < 3
        ]

        if not available_blocks:
            self.game_over = True
            print("--- 全てのブロックを3回ずつ使用しました ---")
            return

        self.current_block_type = random.choice(available_blocks)
        # ▼▼▼ 出現回数をカウントアップ ▼▼▼
        self.block_appearance_count[self.current_block_type] += 1
        self.current_shape_id = 0
        self.cursor_x = self.grid_width // 2
        # 新しいブロックの出現位置を、盤面全体の最高点より少し上にする
        all_heights = [y for x, y in self.placed_blocks]
        self.cursor_y = (max(all_heights) if all_heights else 0) + 10

        # ゲームオーバーチェック
        if not self.is_valid_position(self.cursor_x, self.cursor_y, self.current_shape_id):
            self.game_over = True
            print("--- GAME OVER ---")

    def get_current_rotation_info(self):
        return self.block_definitions[self.current_block_type]['rotations'][self.current_shape_id]

    def get_current_shape_coords(self, x, y):
        shape_info = self.get_current_rotation_info()
        shape_pattern = shape_info['shape']
        anchor_dx, anchor_dy = shape_info['anchor']
        return [(x + dx - anchor_dx, y + dy - anchor_dy) for dx, dy in shape_pattern]

    # def is_valid_position(self, x, y, shape_id_to_check):
    #     temp_shape_info = self.block_definitions[self.current_block_type]['rotations'][shape_id_to_check]
    #     shape_pattern = temp_shape_info['shape']
    #     anchor_dx, anchor_dy = temp_shape_info['anchor']
    #     shape_coords = [(x + dx - anchor_dx, y + dy - anchor_dy) for dx, dy in shape_pattern]
    #     for cx, cy in shape_coords:
    #         if not (0 <= cx < self.grid_width and 0 <= cy < self.grid_height): return False
    #         if (cx, cy) in self.placed_blocks: return False
    #     return True

    def is_valid_position(self, x, y, shape_id_to_check):
        """
        指定した位置が有効かチェックする（上下左右1マスを含めて衝突判定）
        """
        # 1. 配置済みブロックとその上下左右1マスの「進入禁止エリア」を作成する
        forbidden_coords = set()
        # チェックするオフセットを定義（中心、上、下、右、左）
        offsets = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]

        for bx, by in self.placed_blocks.keys():
            for dx, dy in offsets:
                forbidden_coords.add((bx + dx, by + dy))

        # 2. ブロックの各パーツが、進入禁止エリアに入っていないかチェックする
        temp_shape_info = self.block_definitions[self.current_block_type]['rotations'][shape_id_to_check]
        shape_pattern = temp_shape_info['shape']
        anchor_dx, anchor_dy = temp_shape_info['anchor']
        shape_coords = [(x + dx - anchor_dx, y + dy - anchor_dy) for dx, dy in shape_pattern]

        for cx, cy in shape_coords:
            if not (0 <= cx < self.grid_width and 0 <= cy < self.grid_height):
                return False # 画面外
            if (cx, cy) in forbidden_coords:
                return False # 進入禁止エリアと衝突
        
        return True
    def is_stable_position(self):
        """
        ブロックの底面のいずれかが支えられているかチェックする
        """
        x, y = self.cursor_x, self.cursor_y
        shape_coords = self.get_current_shape_coords(x, y)
        
        # 1. まず、ブロック全体がめり込みなく置けるかチェック
        if not self.is_valid_position(x, y, self.current_shape_id):
            return False
        
        # 2. ブロックの各パーツの底が支えられているかチェック
        for sx, sy in shape_coords:
            # 自分自身の他のパーツは「支え」と見なさない
            is_part_of_self = (sx, sy - 2) in shape_coords
            if not is_part_of_self:
                # パーツの真下が床(y=0)であるか、または真下に固定ブロックがある場合
                if sy == 0 or (sx, sy - 2) in self.placed_blocks:
                    return True # 支えられている部分が一つでもあれば安定
            
        return False # どの部分も支えられていない


    def place_block(self):
        if not self.is_stable_position():
            print("不安定な場所には配置できません！")
            return
        shape_coords = self.get_current_shape_coords(self.cursor_x, self.cursor_y)
        self.save_data(self.board_state.copy(), self.current_block_type, self.cursor_x, self.current_shape_id)
        block_color = self.block_definitions[self.current_block_type]['color']
        for cx, cy in shape_coords:
            self.placed_blocks[(cx, cy)] = block_color
        self._update_board_state()
        self.new_block()

    # ▼▼▼▼▼ ここが状態を抽象化する核心部分です ▼▼▼▼▼
    def _update_board_state(self):
        """監視ポイントの高さだけを計算してboard_stateを更新する
        new_heights = []
        for point_x in self.monitoring_points_x:
            # その監視ポイントのX座標にあるブロックの高さをすべてリストアップ
            heights_at_point = [y for x, y in self.placed_blocks if x == point_x]
            # 最大の高さ+1を、そのポイントの高さとする
            height = max(heights_at_point) if heights_at_point else 0
            new_heights.append(height)
        self.board_state = new_heights
        """
        """
        各監視ポイント列の全域をスキャンし、山/谷の形状があればその最高点の高さを記録する
        """
        new_board_state = []

        for point_x in self.monitoring_points_x:
            max_feature_height = -1 # その列で見つかった山/谷の最高点の高さ

            # 各監視ポイントの列を、下から上へスキャン
            for y in range(self.grid_height -1):
                # --- 凸点(ピーク)の判定 ---
                # 条件: (x,y)にブロックがあり、その上と左右が空いている
                is_peak = (
                    (point_x, y) in self.placed_blocks and
                    (point_x, y + 2) not in self.placed_blocks and
                    (point_x, y + 1) not in self.placed_blocks and
                    (point_x - 1, y) not in self.placed_blocks and
                    (point_x - 2, y) not in self.placed_blocks and
                    (point_x + 1, y) not in self.placed_blocks and
                    (point_x + 2, y) not in self.placed_blocks
                )
                if is_peak:
                    max_feature_height = max(max_feature_height, y + 1)

                # --- 鞍点(サドル)の判定 ---
                # 条件: (x,y)は空でひとつ上も空、その下と左右にブロックがある
                is_saddle = (
                    (point_x, y) not in self.placed_blocks and
                    (point_x, y + 1) not in self.placed_blocks and
                    (point_x, y + 2) not in self.placed_blocks and
                    (point_x - 2, y) in self.placed_blocks and
                    (point_x + 2, y) in self.placed_blocks and
                    (point_x - 1, y) in self.placed_blocks and
                    (point_x + 1, y) in self.placed_blocks
                )
                if is_saddle:
                    max_feature_height = max(max_feature_height, y)

            if max_feature_height == -1:
                # その列に山/谷がなかった場合は、-1を記録
                new_board_state.append(max_feature_height)
            else:
                # 山/谷があった場合は、上に障害物がないかチェック
                # 各監視ポイントの列を、下から上へスキャン
                for y in range(max_feature_height + 1, self.grid_height):
                    # 各行で障害物があるかチェック
                    # --- 凸点、鞍点上にブロックがあるかの判定 ---
                    # 条件: (x,y)または、左右いずれかにブロックがある
                    is_obs = (
                        (point_x, y) in self.placed_blocks or
                        (point_x-1, y) in self.placed_blocks or
                        (point_x+1, y) in self.placed_blocks
                    )
                    if is_obs:
                        max_feature_height = -1
                if max_feature_height == -1:
                    # その列に障害物がある場合は、-1を記録
                    new_board_state.append(max_feature_height)
                else:
                    # 障害物がないので最高点の高さを記録、3で割って整数にする
                    new_board_state.append(int(max_feature_height/3))

        self.board_state = new_board_state


    def init_csv(self):
        if not os.path.exists(self.save_path):
            # ▼▼▼ CSVヘッダーを監視ポイントの数に合わせる ▼▼▼
            header = [f'z_{i}' for i in range(len(self.monitoring_points_x))] + ['block_type', 'placed_x', 'rotation_id']
            with open(self.save_path, 'w', newline='') as f: writer = csv.writer(f); writer.writerow(header)

    def save_data(self, state, block_type, x, rot):
        # stateが正しい長さであることを確認 (デバッグ用)
        if len(state) != len(self.monitoring_points_x):
            print(f"警告: 保存するstateの長さ({len(state)})が監視ポイントの数({len(self.monitoring_points_x)})と異なります。")
            return
        row = state + [block_type, int((x - 5)/3), rot]
        with open(self.save_path, 'a', newline='') as f: writer = csv.writer(f); writer.writerow(row)

    def draw(self):
        self.screen.fill((30, 30, 30))
        for (x, y), color in self.placed_blocks.items():
            pixel_x = x * BLOCK_UNIT_SIZE
            pixel_y = SCREEN_HEIGHT - (y + 1) * BLOCK_UNIT_SIZE
            rect = pygame.Rect(pixel_x, pixel_y, BLOCK_UNIT_SIZE, BLOCK_UNIT_SIZE)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (80, 80, 80), rect, 1)

        if hasattr(self, 'current_block_type') and not self.game_over:
            shape_coords = self.get_current_shape_coords(self.cursor_x, self.cursor_y)
            color = self.block_definitions[self.current_block_type]['color']
            is_placeable = self.is_stable_position()
            preview_color = (0, 255, 0, 150) if is_placeable else (*color, 128)
            for x, y in shape_coords:
                pixel_x = x * BLOCK_UNIT_SIZE
                pixel_y = SCREEN_HEIGHT - (y + 1) * BLOCK_UNIT_SIZE
                s = pygame.Surface((BLOCK_UNIT_SIZE, BLOCK_UNIT_SIZE), pygame.SRCALPHA)
                s.fill(preview_color)
                self.screen.blit(s, (pixel_x, pixel_y))

        instructions = ["--- Controls ---", "Arrows: Move", "A: Rot L, D: Rot R", "Space: Place", "R: Reset", "Q: Quit"]
        text = self.font.render(f'Block: {getattr(self, "current_block_type", "N/A")}', True, (255, 255, 255))
        self.screen.blit(text, (10, 10))
        start_y = SCREEN_HEIGHT - len(instructions) * 25 - 20
        for i, line in enumerate(instructions):
            instruction_text = self.font.render(line, True, (200, 200, 200))
            self.screen.blit(instruction_text, (SCREEN_WIDTH - 280, start_y + i * 25))
        
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q: running = False
                    if event.key == pygame.K_r: self.reset_game()
                    if hasattr(self, 'current_block_type') and not self.game_over:
                        next_x, next_y = self.cursor_x, self.cursor_y
                        next_shape_id = self.current_shape_id
                        if event.key == pygame.K_LEFT: next_x -= 1
                        if event.key == pygame.K_RIGHT: next_x += 1
                        if event.key == pygame.K_UP: next_y += 1
                        if event.key == pygame.K_DOWN: next_y -= 1
                        num_rotations = len(self.block_definitions[self.current_block_type]['rotations'])
                        if event.key == pygame.K_d: next_shape_id = (self.current_shape_id + 1) % num_rotations
                        if event.key == pygame.K_a: next_shape_id = (self.current_shape_id - 1 + num_rotations) % num_rotations
                        if self.is_valid_position(next_x, next_y, next_shape_id):
                            self.cursor_x, self.cursor_y = next_x, next_y
                            self.current_shape_id = next_shape_id
                        if event.key == pygame.K_SPACE:
                            self.place_block()
            self.draw()
            self.clock.tick(60)
        pygame.quit()