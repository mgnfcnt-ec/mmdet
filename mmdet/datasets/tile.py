
import os.path as osp
import os
import numpy as np
import cv2

from .builder import DATASETS
from .custom import CustomDataset
from .xml_style import XMLDataset

import xml.etree.ElementTree as ET
import misc_utils  # 使用 pip install utils-misc 安装
from mmcv import list_from_file
from mmcv.utils import print_log

from collections import defaultdict, OrderedDict

from mmdet.core import eval_map

@DATASETS.register_module()
class TileDataset(CustomDataset):

    CLASSES = ('_bkg', 'edge', 'corner', 'whitespot', 'lightblock', 'darkblock', 'aperture')

    def __init__(self, min_size=None, **kwargs):
        super(TileDataset, self).__init__(**kwargs)
        # self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size

    def load_annotations(self, ann_file):
        datasplit = set(list_from_file(ann_file))
        annos = misc_utils.load_json(osp.join(osp.dirname(ann_file), 'train_annos.json'))
        data_infos = []

        image_names = set()
        metas = defaultdict(dict)
        bboxes = defaultdict(list)
        labels = defaultdict(list)

        for i, line in enumerate(annos):
            misc_utils.progress_bar(i, len(annos), 'Load Anno...')
            """
                line =
                    {'name': '223_89_t20201125085855802_CAM3.jpg',
                    'image_height': 3500,
                    'image_width': 4096,
                    'category': 4,
                    'bbox': [1702.79, 2826.53, 1730.79, 2844.53]}
            """
            name = line['name']
            img_id = misc_utils.get_file_name(name)
            height = line['image_height']
            width = line['image_width']
            label = line['category']
            bbox = line['bbox']  # xyxy
            filename = f'train_imgs/{name}'

            if name not in datasplit:  # 意思是不在训练集里就跳过
                continue

            image_names.add(name)
            bboxes[name].append(bbox)
            labels[name].append(label)
            metas[name] = dict(id=img_id, filename=filename, width=width, height=height)

        for i, name in enumerate(image_names):
            ann = dict(
                bboxes=np.array(bboxes[name]).astype(np.float32),
                labels=np.array(labels[name]).astype(np.int64)
            )
            meta = metas[name]
            meta.update({'ann': ann})

            data_infos.append(meta)

        return data_infos

@DATASETS.register_module()
class TileCropDataset(XMLDataset):

    CLASSES = ('edge', 'corner', 'whitespot', 'lightblock', 'darkblock', 'aperture')
    # CLASSES = ('边异常', '角异常', '白色点瑕疵', '浅色块瑕疵', '深色点块瑕疵', '光圈瑕疵')
    def __init__(self, **kwargs):
        super(TileCropDataset, self).__init__(**kwargs)

    def _non_max_suppress(self, predicts_dict, totalImage, threshold=0.01):
        """which is used by TileCropDataset's evaluation.
        """
        ret = []
        for i in range(totalImage):
            bboxPerC = []

            for j in range(len(self.CLASSES)):
                bbox_array = predicts_dict[i][j]

                x1, y1, x2, y2, scores = bbox_array[:, 0], bbox_array[:, 1], bbox_array[:, 2], bbox_array[:,
                                                                                               3], bbox_array[:, 4]
                areas = (x2 - x1 + 1) * (y2 - y1 + 1)
                order = scores.argsort()[::-1]
                keep = []  # 用来存放最终保留的bbx的索引信息

                ## 依次从按confidence从高到低遍历bbx，移除所有与该矩形框的IOU值大于threshold的矩形框
                while order.size > 0:
                    ii = order[0]
                    keep.append(ii)  # 保留当前最大confidence对应的bbx索引

                    ## 获取所有与当前bbx的交集对应的左上角和右下角坐标，并计算IOU（注意这里是同时计算一个bbx与其他所有bbx的IOU）
                    xx1 = np.maximum(x1[ii], x1[order[1:]])
                    # 当order.size=1时，下面的计算结果都为np.array([]),不影响最终结果
                    yy1 = np.maximum(y1[ii], y1[order[1:]])
                    xx2 = np.minimum(x2[ii], x2[order[1:]])
                    yy2 = np.minimum(y2[ii], y2[order[1:]])
                    inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
                    iou = inter / (areas[ii] + areas[order[1:]] - inter)
                    indexs = np.where(iou <= threshold)[0] + 1  # 获取保留下来的索引(因为没有计算与自身的IOU，所以索引相差１，需要加上)
                    order = order[indexs]  # 更新保留下来的索引

                bboxPerC.append(bbox_array[keep])
            ret.append(bboxPerC)
        return ret

    def _judgeWhetherExistBBox(self, resultPerImage):
        for j in range(len(self.CLASSES)):
            if resultPerImage[j].shape[0] > 0:
                return True
        return False

    def _plotResult(self,BBoxPerImage,GT_PerImage,  FileName, FileNameNew):
        img = cv2.imread(FileName)

        for j in range(len(self.CLASSES)):
            for goal in BBoxPerImage[j]:
                cv2.rectangle(img, (int(goal[0]), int(goal[1])), (int(goal[2]), int(goal[3])),
                              (0, 122, 122), 1)
                cv2.putText(img, str(j) + ' ' + str(goal[4]), (int(goal[0]), int(goal[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        for i, perGT in enumerate(GT_PerImage):
            cv2.rectangle(img, (int(perGT[0]), int(perGT[1])), (int(perGT[2]), int(perGT[3])), (255, 0, 0), 1)
        cv2.imwrite(FileNameNew, img)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.1,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'mAP_mergeCrop']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')


        if metric == 'mAP':
            """
                Defalut metric,which just evaluates the seperate crop images.
            """
            annotations = [self.get_ann_info(i) for i in range(len(self))]
            eval_results = OrderedDict()
            iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)

        elif metric == 'mAP_mergeCrop':
            """
                A custom metric,which merges the crop images into the original.And it'll evaluate the detection for the original images.
                The score of the metric is equivilent of Guangdong Competition's.
                Specifically, score = 0.2 * ACC + 0.8 * mAP
                
                ACC: it is the classification index of defect or no defect, which inspects the ability of defect detection.
                
                Map: calculate the map value of defects according to PascalVOC evaluation standard.
                The map was calculated at the cross union ratio (IOU) of the detection frame and the real frame at the thresholds of 0.1, 0.3 and 0.5. 
                And the final map was the average of the three values.
            """
            # STEP1:
            # merge the crop images

            results_merge = {}

            for i,perCrop in enumerate(self.data_infos):
                idx, _, dx, dy, __, ___, ____ = perCrop['id'].split('_')
                idx = int(idx)
                dx, dy = float(dx), float(dy)
                for j in range(len(self.CLASSES)):
                    results[i][j][:, 0:4:2] += dx
                    results[i][j][:, 1:4:2] += dy

                if idx in results_merge.keys():
                    for j in range(len(self.CLASSES)):
                        results_merge[idx][j] = np.append(results_merge[idx][j], results[i][j], axis=0)
                else:
                    results_merge[idx] = results[i]

            totalImage = len(results_merge)

            # STEP2:
            # NMS for the numpy array
            results_new = self._non_max_suppress(results_merge, totalImage, 0.02)

            # STEP3:
            # load the original image GT. And count how many image are detected correctly, even more plot the result.

            annotations = []
            cnt_DetectTrue = 0
            jpg_root = osp.join(osp.dirname(osp.dirname(self.img_prefix)),'voc-ori-val','JPEGImages')
            for i,perImageName in enumerate(os.listdir(jpg_root)):
                # control the Image numbers, consistent with slice part
                if i == totalImage:
                    break

                if self._judgeWhetherExistBBox(results_new[i]) == True:
                    cnt_DetectTrue += 1

                # read the original
                xmlfile = osp.join(osp.dirname(jpg_root), 'Annotations', perImageName[:-4] + '.xml')
                with open(xmlfile, "r", encoding='UTF-8') as in_file:
                    tree = ET.parse(in_file)
                    root = tree.getroot()
                    size = root.find('size')
                    sizew = int(size.find('width').text)
                    sizeh = int(size.find('height').text)

                fh1 = open(osp.join(osp.dirname(jpg_root), 'labels', perImageName[:-4] + '.txt'), "r")

                boxes = []
                labels = []
                for line in fh1.readlines():
                    splitLine = line.split(" ")
                    labels.append(int(splitLine[0]))  # class
                    x = float(splitLine[1])  # confidence
                    y = float(splitLine[2])
                    w = float(splitLine[3])
                    h = float(splitLine[4])
                    x1 = (x - w / 2.0) * sizew
                    y1 = (y - h / 2.0) * sizeh
                    x2 = (x + w / 2.0) * sizew
                    y2 = (y + h / 2.0) * sizeh
                    boxes.append([x1, y1, x2, y2])

                annotations.append({'bboxes': np.array(boxes), 'labels':np.array(labels)})

                # self._plotResult(results_new[i], boxes, osp.join(jpg_root, perImageName), "res-show/" + perImageName)

            # STEP4: evaluate
            eval_results = OrderedDict()
            iou_thrs = [0.1, 0.3, 0.5]
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    results_new,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results['ACC'] = cnt_DetectTrue / totalImage
            eval_results['FinalScore'] = eval_results['ACC'] * 0.2 + eval_results['mAP'] * 0.8

        return eval_results
