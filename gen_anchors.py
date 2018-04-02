"""
Created on Feb 20, 2017

@author: jumabek
"""
import argparse
import numpy as np
import os
import random


def iou(x, centroids):
    ious = []
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x

        if c_w >= w and c_h >= h:
            intersection, union = w * h, c_w * c_h
        elif c_w >= w and c_h <= h:
            intersection, union = w * c_h, w * h + (c_w-w) * c_h
        elif c_w <= w and c_h >= h:
            intersection, union = c_w * h, w * h + c_w * (c_h-h)
        else:
            intersection, union = c_w * c_h, w * h

        ious.append(intersection / union)

    return np.array(ious)


def avg_iou(annotation_dims, centroids):
    """

    :param annotation_dims:
    :param centroids:
    :return:
    """
    total = 0
    for annotation in annotation_dims:
        # note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective,
        # but I am too lazy
        total += max(iou(annotation, centroids))

    n_annotations, _ = annotation_dims.shape
    return total / n_annotations


def write_anchors_to_file(centroids, annotation_dims, anchor_file):
    anchors = centroids.copy()
    print(anchors.shape)

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])
    with open(anchor_file, 'w') as f:
        for i in sorted_indices[:-1]:
            f.write('%0.2f,%0.2f, ' % (anchors[i, 0], anchors[i, 1]))

        # there should not be comma after last anchor, that's why
        f.write('%0.2f,%0.2f\n' % (anchors[sorted_indices[-1:], 0], anchors[sorted_indices[-1:], 1]))

        f.write('%f\n' % (avg_iou(annotation_dims, centroids)))
        print()


def k_means(annotation_dims, centroids):
    num_annotations = annotation_dims.shape[0]
    prev_assignments = None

    count = 0
    while True:
        distances = []
        for i in range(num_annotations):
            distance = 1 - iou(annotation_dims[i], centroids)
            distances.append(distance)

        print("iter {}: mean = {}".format(count, np.mean(distances)))

        # assign samples to centroids
        # distances have a shape of (n_annotations, k)
        # assign the annotations to the closest cluster
        new_assignments = np.argmin(distances, axis=1)

        if (new_assignments == prev_assignments).all():
            print('Centroids have not changed since the last iteration. Finished searching!')
            print("Centroids = ", centroids)
            return centroids, annotation_dims

        # calculate new centroids
        centroid_annotations = [list() for _ in range(len(centroids))]
        for cluster, annotation in zip(new_assignments, annotation_dims):
            centroid_annotations[cluster].append(annotation)

        for i, annotations in enumerate(centroid_annotations):
            centroids[i] = np.mean(annotations, axis=0)

        prev_assignments = np.copy(new_assignments)
        count += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-filelist', default='\\path\\to\\voc\\filelist\\train.txt',
                        help='path to filelist\n')
    parser.add_argument('-output_dir', default='generated_anchors/anchors', type=str,
                        help='Output anchor directory\n')
    parser.add_argument('-num_clusters', default=0, type=int,
                        help='number of clusters\n')

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    annotation_dims = []

    with open(args.filelist) as image_list:
        for image_path in image_list.read().splitlines():

            image_path = image_path.replace('images', 'labels')
            image_path = image_path.replace('img1', 'labels')
            image_path = image_path.replace('JPEGImages', 'labels')
            image_path = image_path.replace('.jpg', '.txt')
            image_path = image_path.replace('.png', '.txt')

            annotation_path = image_path
            print(annotation_path)
            annotations = open(annotation_path)
            for line in annotations.read().splitlines():
                w, h = line.split()[3:]
                annotation_dims.append(list(map(float, (w, h))))
        annotation_dims = np.array(annotation_dims)
  
    if args.num_clusters == 0:
        for num_clusters in range(1, 11):
            anchor_file = os.path.join(args.output_dir, 'anchors%d.txt' % num_clusters)

            indices = [random.randrange(annotation_dims.shape[0]) for _ in range(num_clusters)]
            centroids = annotation_dims[indices]
            centroids, annotation_dims = k_means(annotation_dims, centroids)
            write_anchors_to_file(centroids, annotation_dims, anchor_file)
    else:
        anchor_file = os.path.join(args.output_dir, 'anchors%d.txt' % args.num_clusters)
        indices = [random.randrange(annotation_dims.shape[0]) for _ in range(args.num_clusters)]
        centroids = annotation_dims[indices]
        centroids, annotation_dims = k_means(annotation_dims, centroids)
        write_anchors_to_file(centroids, annotation_dims, anchor_file)


if __name__ == "__main__":
    main()
