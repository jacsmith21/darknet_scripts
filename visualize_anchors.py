import cv2
import os
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-anchor_dir', default='generated_anchors/voc-anchors',
                        help='path to anchors\n', )

    args = parser.parse_args()

    print("anchors list you provided: {}".format(args.anchor_dir))

    h, w = (416, 416)
    stride = 32

    cv2.namedWindow('Image')
    cv2.moveWindow('Image', 100, 100)

    colors = [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (55, 0, 0), (255, 55, 0), (0, 55, 0),
              (0, 0, 25), (0, 255, 55)]

    anchor_files = [f for f in os.listdir(args.anchor_dir) if (os.path.join(args.anchor_dir, f)).endswith('.txt')]
    for anchor_file in anchor_files:
        blank_image = np.zeros((h, w, 3), np.uint8)

        f = open(os.path.join(args.anchor_dir, anchor_file))
        line = f.readline().rstrip('\n')

        anchors = line.split(', ')

        filename = os.path.join(args.anchor_dir, anchor_file).replace('.txt', '.png')

        print(filename)

        stride_h = 10
        stride_w = 3
        if 'caltech' in filename:
            stride_w = 25
            stride_h = 10

        for i in range(len(anchors)):
            (w, h) = list(map(float, anchors[i].split(',')))

            w = int(w * stride)
            h = int(h * stride)
            print(w, h)
            cv2.rectangle(blank_image, (10 + i * stride_w, 10 + i * stride_h), (w, h), colors[i])

            cv2.imwrite(filename, blank_image)


if __name__ == "__main__":
    main()
