import cv2


def draw_bbox(img, bboxes, color=(0, 255, 0)):
    for bbox in bboxes:
        img = cv2.rectangle(
            img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2
        )
    return img


def draw_skeleton(
    img, keypoints, scores, metainfo, kpt_thr=0.5, radius=2, line_width=2
):
    keypoint_info = metainfo["keypoint_info"]
    skeleton_info = metainfo["skeleton_info"]

    if len(keypoints.shape) == 2:
        keypoints = keypoints[None, :, :]
        scores = scores[None, :, :]

    num_instance = keypoints.shape[0]
    for i in range(num_instance):
        img = draw_mmpose(
            img,
            keypoints[i],
            scores[i],
            keypoint_info,
            skeleton_info,
            kpt_thr,
            radius,
            line_width,
        )
    return img


def draw_mmpose(
    img,
    keypoints,
    scores,
    keypoint_info,
    skeleton_info,
    kpt_thr=0.5,
    radius=2,
    line_width=2,
):
    assert len(keypoints.shape) == 2

    vis_kpt = [s >= kpt_thr for s in scores]

    link_dict = {}
    for i, kpt_info in keypoint_info.items():
        link_dict[kpt_info["name"]] = kpt_info["id"]

    for i, ske_info in skeleton_info.items():
        link = ske_info["link"]
        link_color = ske_info["color"]
        pt0, pt1 = link_dict[link[0]], link_dict[link[1]]

        if vis_kpt[pt0] and vis_kpt[pt1]:
            kpt0 = keypoints[pt0]
            kpt1 = keypoints[pt1]

            img = cv2.line(
                img,
                (int(kpt0[0]), int(kpt0[1])),
                (int(kpt1[0]), int(kpt1[1])),
                link_color[::-1],
                thickness=line_width,
            )

    stroke_color = (255, 255, 255)  # White
    fill_color = (204, 51, 51)  # Crimson Red
    for i, kpt_info in keypoint_info.items():
        kpt = keypoints[i]

        if vis_kpt[i]:
            center = (int(kpt[0]), int(kpt[1]))

            # Draw outer circle (stroke)
            img = cv2.circle(
                img, center, int(radius + line_width), stroke_color[::-1], -1
            )

            # Draw inner circle (fill)
            img = cv2.circle(img, center, int(radius), fill_color[::-1], -1)

    return img
