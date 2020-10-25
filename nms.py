import numpy as np

'''
目标检测中常用到NMS，在faster R-CNN中，每一个bounding box都有一个打分，NMS实现逻辑是：

    1，按打分最高到最低将BBox排序 ，例如：A B C D E F

    2，A的分数最高，保留。从B-E与A分别求重叠率IoU，假设B、D与A的IoU大于阈值，那么B和D可以认为是重复标记去除

    3，余下C E F，重复前面两步。

'''
def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]  # bbox打分
    print(x1)

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 打分从大到小排列，取index
    order = scores.argsort()[::-1]
    # keep为最后保留的边框
    keep = []
    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)
        # 计算窗口i与其他所有窗口的交叠部分的面积

        xx1 = max(x1[i], x1[order[1:]])
        yy1 = max(y1[i], y1[order[1:]])
        xx2 = min(x2[i], x2[order[1:]])
        yy2 = min(y2[i], y2[order[1:]])

        w = max(0.0, xx2 - xx1 + 1)
        h = max(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= thresh)[0]
        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        print('inds',inds,inds+1)
        order = order[inds + 1]  # inds 的第一个索引对应order的第二个索引

    return keep


if __name__=='__main__':
    print(py_cpu_nms(np.array([[661, 27, 679, 47, 0.8], [662, 27, 682, 47, 0.9]]),0.83))