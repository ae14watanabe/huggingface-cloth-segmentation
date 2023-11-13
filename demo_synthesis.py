import gradio as gr
from synthesis import generate_upper_cloth_mask, find_bounding_quad
import cv2
import numpy as np
from process import apply_transform, load_seg_model
from PIL import Image


def process_images(woman_tshirts, grey_muji_mask, logo_located):
    # ここで画像を処理し、合成画像を生成する
    # 上記のスクリプトの処理を適応する
    # 結果の画像を返す
    checkpoint_path = "model/cloth_segm.pth"
    net = load_seg_model(checkpoint_path)
    device = "cpu"

    woman_tshirts_mask = generate_upper_cloth_mask(
        woman_tshirts.convert("RGB"), net, device
    )
    if grey_muji_mask.mode != "L":
        grey_muji_mask = grey_muji_mask.convert("L")

    print(grey_muji_mask)
    print(type(grey_muji_mask))
    print(np.array(grey_muji_mask).shape)

    print("logo_located info:")
    print(logo_located)
    print(type(logo_located))
    print(logo_located.mode)
    print(np.array(logo_located).shape)

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

    # それぞれRGBに変換して返す
    return (
        cv2.cvtColor(composite_image_cv, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(woman_tshirts_mask_cv, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(muji_composite, cv2.COLOR_BGR2RGB),
    )


# Gradioインターフェース
iface = gr.Interface(
    fn=process_images,
    inputs=[
        gr.Image(type="pil", label="Wearing T-Shirts image"),
        gr.Image(type="pil", label="Hiraoki T-shirts Mask"),
        gr.Image(type="pil", label="Logo Located", image_mode="RGBA"),
    ],
    outputs=[
        gr.Image(type="pil", label="Synthesis image"),
        gr.Image(type="pil", label="Segmented t-shirts mask image"),
        gr.Image(type="pil", label="Logo located on t-shirts mask image"),
    ],
)


# デモの起動
iface.launch()
