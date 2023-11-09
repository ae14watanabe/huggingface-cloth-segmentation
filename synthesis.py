import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import torch
from process import apply_transform, load_seg_model
import torch.nn.functional as F


def generate_upper_cloth_mask(input_image, net, device="cpu"):
    # img = Image.open(input_image).convert('RGB')
    img = input_image
    img_size = img.size
    img = img.resize((768, 768), Image.BICUBIC)
    image_tensor = apply_transform(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    with torch.no_grad():
        output_tensor = net(image_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

    alpha_mask = (output_arr == 1).astype(np.uint8) * 255
    alpha_mask = alpha_mask[0]  # Selecting the first channel to make it 2D
    alpha_mask_img = Image.fromarray(alpha_mask, mode="L")
    alpha_mask_img = alpha_mask_img.resize(img_size, Image.BICUBIC)
    return alpha_mask_img


# 輪郭を見つけて四角形の頂点を返す関数
def find_bounding_quad(img_mask):
    contours, _ = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 最大の輪郭を取得する
    c = max(contours, key=cv2.contourArea)
    # 輪郭のバウンディングボックスを取得する
    x, y, w, h = cv2.boundingRect(c)
    # バウンディングボックスの四隅の座標を返す
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])


if __name__ == "__main__":
    checkpoint_path = "model/cloth_segm.pth"
    net = load_seg_model(checkpoint_path)
    device = "cpu"

    # 画像パス
    woman_tshirts_path = "prepared_images/woman_tshirts.png"
    grey_muji_mask_path = "prepared_images/grey_muji_mask.png"
    logo_located_path = "prepared_images/surisuri_located.png"

    # 画像を読み込む
    woman_tshirts = Image.open(woman_tshirts_path)
    woman_tshirts_mask = generate_upper_cloth_mask(
        woman_tshirts.convert("RGB"), net, device
    )

    grey_muji_mask = Image.open(grey_muji_mask_path)
    logo_located = Image.open(logo_located_path)

    # PIL画像をOpenCV画像に変換
    woman_tshirts_cv = cv2.cvtColor(np.array(woman_tshirts), cv2.COLOR_RGB2BGR)
    woman_tshirts_mask_cv = cv2.cvtColor(
        np.array(woman_tshirts_mask), cv2.COLOR_GRAY2BGR
    )
    grey_muji_mask_cv = cv2.cvtColor(np.array(grey_muji_mask), cv2.COLOR_GRAY2BGR)
    logo_located_cv = cv2.cvtColor(np.array(logo_located), cv2.COLOR_RGBA2BGRA)

    # マスクから二値画像を生成
    _, woman_tshirts_mask_bin = cv2.threshold(
        cv2.cvtColor(woman_tshirts_mask_cv, cv2.COLOR_BGR2GRAY),
        1,
        255,
        cv2.THRESH_BINARY,
    )
    _, grey_muji_mask_bin = cv2.threshold(
        cv2.cvtColor(grey_muji_mask_cv, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY
    )

    # 頂点の検出
    woman_tshirts_corners = find_bounding_quad(woman_tshirts_mask_bin)
    grey_muji_corners = find_bounding_quad(grey_muji_mask_bin)

    # 頂点の順番を左上、右上、右下、左下に整理
    woman_tshirts_corners_sorted = woman_tshirts_corners[
        np.argsort(woman_tshirts_corners[:, 0])
    ]
    grey_muji_corners_sorted = grey_muji_corners[np.argsort(grey_muji_corners[:, 0])]

    # ホモグラフィ行列の計算
    H, _ = cv2.findHomography(grey_muji_corners_sorted, woman_tshirts_corners_sorted)

    # ロゴをホモグラフィ行列を使って変形させる
    logo_warped = cv2.warpPerspective(
        logo_located_cv, H, (woman_tshirts_cv.shape[1], woman_tshirts_cv.shape[0])
    )

    # ロゴ画像のアルファチャネルをマスクとして取得する
    alpha_mask = logo_warped[:, :, 3] / 255.0
    inverse_alpha_mask = 1.0 - alpha_mask

    # ロゴ画像の色情報のみを取得する（アルファチャネルを除外）
    logo_color = logo_warped[:, :, :3]

    # Tシャツ画像とロゴ画像を合成する
    composite_image_cv = woman_tshirts_cv.copy()
    for c in range(0, 3):
        composite_image_cv[:, :, c] = (
            alpha_mask * logo_color[:, :, c]
            + inverse_alpha_mask * woman_tshirts_cv[:, :, c]
        )

    # grey_muji_mask上にlogo_locatedを重ねる
    # この操作は、アルファチャネルが存在する場合にのみ意味をなす
    # logo_locatedのアルファチャネルを取得
    logo_alpha = logo_located_cv[:, :, 3] / 255.0
    inverse_logo_alpha = 1.0 - logo_alpha

    # ロゴの色情報のみを取得（アルファチャネルを除外）
    logo_color_only = logo_located_cv[:, :, :3]

    # grey_muji_mask上にロゴを重ねる
    muji_composite = grey_muji_mask_cv.copy()
    for c in range(0, 3):
        muji_composite[:, :, c] = (
            logo_alpha * logo_color_only[:, :, c]
            + inverse_logo_alpha * muji_composite[:, :, c]
        )

    # 合成画像とオリジナルのTシャツ画像、そしてマスク上にロゴを重ねた画像を横並びで表示
    plt.figure(figsize=(18, 6))

    # オリジナルのTシャツ画像
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(woman_tshirts_cv, cv2.COLOR_BGR2RGB))
    plt.title("Original T-Shirt")
    plt.axis("off")

    # マスク上にロゴを重ねた画像
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(muji_composite, cv2.COLOR_BGR2RGB))
    plt.title("Logo on Muji Mask")
    plt.axis("off")

    # 合成された画像
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(composite_image_cv, cv2.COLOR_BGR2RGB))
    plt.title("Composite Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
