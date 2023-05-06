import numpy as np
import matplotlib.pyplot as plt


def nms(bboxes, scores, thresh, mode="Union"):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def plot_bbox(dets, title, c='k'):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    plt.plot([x1, x2], [y1, y1], c)
    plt.plot([x1, x1], [y1, y2], c)
    plt.plot([x1, x2], [y2, y2], c)
    plt.plot([x2, x2], [y1, y2], c)
    plt.title(title)


if __name__ == '__main__':
    bboxes = np.array([[100, 100, 210, 210],
                      [200, 250, 320, 420],
                      [230, 220, 340, 330],
                      [150, 140, 260, 250],
                      [230, 240, 325, 330],
                      [220, 230, 315, 340]])
    scores = np.array([0.7, 0.8, 0.9, 0.7, 0.8, 0.9])
    
    fig = plt.figure(1)
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    plt.sca(ax1)
    plot_bbox(bboxes, 'Before NMS',  'k')  # before nms

    keep = nms(bboxes, scores, thresh=0.6)

    plt.sca(ax2)
    plot_bbox(bboxes[keep], 'After NMS', 'r')  # after nms

    plt.savefig('./nms.jpg')
    plt.show()
