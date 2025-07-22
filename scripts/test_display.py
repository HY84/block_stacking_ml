# scripts/test_display.py
import pygame
import os

# 対策2の設定も入れておく
os.environ['SDL_VIDEODRIVER'] = 'x11'

pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Display Test")

print("ウィンドウを表示します。5秒後に自動で閉じます。")

running = True
# 5秒間だけループ
start_time = pygame.time.get_ticks()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    if pygame.time.get_ticks() - start_time > 5000: # 5秒経過したら
        running = False

    screen.fill((255, 0, 0)) # 赤い画面
    pygame.display.flip()

pygame.quit()
print("テストプログラムが正常に終了しました。")