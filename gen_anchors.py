"""
Created on Feb 20, 2017

@author: jumabek
"""
import argparse
import numpy as np
import os
import random

width_in_cfg_file = 416.
height_in_cfg_file = 416.


def iou(x, centroids):
    similarities = []
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w >= w and c_h >= h:
            similarity = w*h / (c_w*c_h)
        elif c_w >= w and c_h <= h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity)  # will become (k,) shape
    return np.array(similarities) 


def avg_iou(annotation_dims, centroids):
    n_annotations, _ = annotation_dims.shape
    total = 0.
    for i in range(annotation_dims.shape[0]):
        # note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective,
        # but I am too lazy
        total += max(iou(annotation_dims[i], centroids))
    return total / n_annotations


def write_anchors_to_file(centroids, annotation_dims, anchor_file):
    f = open(anchor_file, 'w')
    
    anchors = centroids.copy()
    print(anchors.shape)

    for i in range(anchors.shape[0]):
        anchors[i][0] *= width_in_cfg_file / 32
        anchors[i][1] *= height_in_cfg_file / 32

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])
        
    for i in sorted_indices[:-1]:
        f.write('%0.2f,%0.2f, ' % (anchors[i, 0], anchors[i, 1]))

    # there should not be comma after last anchor, that's why
    f.write('%0.2f,%0.2f\n' % (anchors[sorted_indices[-1:], 0], anchors[sorted_indices[-1:], 1]))
    
    f.write('%f\n' % (avg_iou(annotation_dims, centroids)))
    print()


def k_means(annotation_dims, centroids, anchor_file):
    num_annotations = annotation_dims.shape[0]
    k, _ = centroids.shape
    prev_assignments = None
    old_distances = np.zeros((num_annotations, k))

    count = 0
    while True:
        distances = []
        for i in range(num_annotations):
            distance = 1 - iou(annotation_dims[i], centroids)
            distances.append(distance)

        print("iter {}: dists = {}, diff = {}".format(count, np.sum(distances), np.sum(old_distances-distances)))

        # assign samples to centroids
        assignments = np.argmin(distances, axis=1)

        if (assignments == prev_assignments).all():
            print("Centroids = ", centroids)
            write_anchors_to_file(centroids, annotation_dims, anchor_file)
            return

        # calculate new centroids
        centroid_sums = np.zeros_like(centroids)
        for i in range(num_annotations):
            centroid_sums[assignments[i]] += annotation_dims[i]
        for j in range(k):
            centroids[j] = centroid_sums[j]/(np.sum(assignments == j))

        prev_assignments = np.copy(assignments)
        old_distances = np.copy(distances)

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
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    image_list = open(args.filelist)
  
    lines = [image_path.rstrip('\n') for image_path in image_list.readlines()]
    
    annotation_dims = []

    for image_path in lines:
                    
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
            k_means(annotation_dims, centroids, anchor_file)
            print('centroids.shape', centroids.shape)
    else:
        anchor_file = os.path.join(args.output_dir, 'anchors%d.txt' % args.num_clusters)
        indices = [random.randrange(annotation_dims.shape[0]) for _ in range(args.num_clusters)]
        centroids = annotation_dims[indices]
        k_means(annotation_dims, centroids, anchor_file)
        print('centroids.shape', centroids.shape)


if __name__ == "__main__":
    main()
