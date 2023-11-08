import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


woman_tshirts_path = "prepared_images/woman_tshirts.png"
woman_tshirts_mask_path = "woman_tshirts_mask.png"
grey_muji_mask_path = "prepared_images/grey_muji_mask.png"
surisuri_located_path = "prepared_images/surisuri_located.png"

# 画像を読み込む
woman_tshirts = Image.open(woman_tshirts_path)
woman_tshirts_mask = Image.open(woman_tshirts_mask_path)
grey_muji_mask = Image.open(grey_muji_mask_path)
surisuri_located = Image.open(surisuri_located_path)

woman_tshirts_cv = cv2.cvtColor(np.array(woman_tshirts), cv2.COLOR_RGB2BGR)
woman_tshirts_mask_cv = cv2.cvtColor(np.array(woman_tshirts_mask), cv2.COLOR_GRAY2BGR)
grey_muji_mask_cv = cv2.cvtColor(np.array(grey_muji_mask), cv2.COLOR_GRAY2BGR)
surisuri_located_cv = cv2.cvtColor(np.array(surisuri_located), cv2.COLOR_RGBA2BGRA)


# 特徴点検出器を作成
# sift = cv2.SIFT_create()
# akaze = cv2.AKAZE_create()
orb = cv2.ORB_create()

# Tシャツのマスクを適用して特徴点を検出
tshirt_area = cv2.bitwise_and(woman_tshirts_cv, woman_tshirts_mask_cv)
_, tshirt_mask = cv2.threshold(
    cv2.cvtColor(tshirt_area, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY
)

keypoints1, descriptors1 = orb.detectAndCompute(tshirt_area, tshirt_mask)

# 基準のTシャツについても同様に特徴点を検出
_, grey_muji_mask_bin = cv2.threshold(
    cv2.cvtColor(grey_muji_mask_cv, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY
)

keypoints2, descriptors2 = orb.detectAndCompute(grey_muji_mask_cv, grey_muji_mask_bin)

# 特徴点を描画して確認
tshirt_keypoints = cv2.drawKeypoints(tshirt_area, keypoints1, None)
muji_keypoints = cv2.drawKeypoints(grey_muji_mask_cv, keypoints2, None)

# OpenCVの画像をmatplotlibで表示できるように変換
tshirt_keypoints_rgb = cv2.cvtColor(tshirt_keypoints, cv2.COLOR_BGR2RGB)
muji_keypoints_rgb = cv2.cvtColor(muji_keypoints, cv2.COLOR_BGR2RGB)

# 特徴点間のマッチングを行う
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# マッチング結果を距離でソート（良いマッチ順）
matches = sorted(matches, key=lambda x: x.distance)

# マッチング結果を描画
matched_img = cv2.drawMatches(
    tshirt_area,
    keypoints1,
    grey_muji_mask_cv,
    keypoints2,
    matches[:50],  # トップ50のマッチのみ描画
    None,
    flags=2,
)

# 特徴点を描画した画像を表示する
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.imshow(tshirt_keypoints_rgb)
# plt.title("Woman T-Shirts Keypoints")
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.imshow(muji_keypoints_rgb)
# plt.title("Grey Muji T-Shirt Keypoints")
# plt.axis("off")

# plt.tight_layout()
# plt.show()
# OpenCVの画像をmatplotlibで表示できるように変換
matched_img_rgb = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)

# マッチング結果を表示する
plt.figure(figsize=(12, 6))
plt.imshow(matched_img_rgb)
plt.title("Feature Points Matching")
plt.axis("off")
plt.show()

# ホモグラフィ行列を計算するための対応点を取得
points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# ホモグラフィ行列を計算（RANSACを使用して外れ値を排除）
H, _ = cv2.findHomography(points2, points1, cv2.RANSAC)

# ロゴをホモグラフィ行列を使って変形させる
logo_warped = cv2.warpPerspective(
    surisuri_located_cv, H, (tshirt_area.shape[1], tshirt_area.shape[0])
)

# ロゴの変形後の画像を表示する
logo_warped_rgba = cv2.cvtColor(logo_warped, cv2.COLOR_BGRA2RGBA)

# plt.figure(figsize=(6, 6))
# plt.imshow(logo_warped_rgba)
# plt.title("Warped Logo")
# plt.axis("off")
# plt.show()
