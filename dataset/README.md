# 原始数据位置

原始数据保留在课程下载目录，本项目不修改原图。运行时用 `--data-root` 指向含 `hazelnut/`、`zipper/` 和 `image_anomaly_labels.json` 的目录。

当前本机位置为 `../../DM_2026_Dataset/Image_Anomaly_Detection/Image_Anomaly_Detection`（相对本仓库根目录）。也可以将相同结构放在本目录后传入 `--data-root dataset`。

测试图片文件名只在各自类别中唯一；必须用“类别＋相对路径”对齐标签。原始图片和标签不纳入 Git。
