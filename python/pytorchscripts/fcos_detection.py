import torch
from torchvision.ops import nms
from .shape import Shape


class FCOS_Detection(object):

    def __init__(self, labels, score_threshold=None, nms_threshold=0.5, img_size=[384, 640], joints=None):
        self.labels = labels
        self.index_to_label = {i: labels[i] for i in range(len(labels))}
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.strides = (8, 16, 32, 64, 128)
        self.nms_pre = 1000
        self.img_size = img_size
        self.joints = joints
        self.max_joints = max(joints)
        self.joints_dim = self.max_joints * 2

    def __call__(self, cls_scores, vertex_preds, centernesses, occlusions):
        mlvl_vertexes = []
        mlvl_scores = []
        mlvl_centerness = []
        mlvl_occlusions = []
        detected_shapes = []
        num_levels = len(cls_scores)
        num_classes = cls_scores[0].size(1)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, vertex_preds[0].dtype, vertex_preds[0].device)
        cls_score_list = [cls_scores[i][0].detach() for i in range(num_levels)]
        vertex_pred_list = [vertex_preds[i][0].detach() for i in range(num_levels)]
        centerness_list = [centernesses[i][0].detach() for i in range(num_levels)]
        occlusion_list = [occlusions[i][0].detach() for i in range(num_levels)]
        for cls_score, vertex_pred, centerness, occlusion, points in zip(cls_score_list, vertex_pred_list, centerness_list, occlusion_list, mlvl_points):
            scores = cls_score.permute(1, 2, 0).reshape(-1, num_classes).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            vertex_pred = vertex_pred.permute(1, 2, 0).reshape(-1, self.joints_dim)
            occlusion = occlusion.permute(1, 2, 0).reshape(-1, self.joints_dim).sigmoid()
            if self.nms_pre > 0 and scores.shape[0] > self.nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(self.nms_pre)
                points = points[topk_inds, :]
                vertex_pred = vertex_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
                occlusion = occlusion[topk_inds, :]
            points = points.repeat(1, self.max_joints)
            vertexes = vertex_pred + points
            mlvl_vertexes.append(vertexes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_occlusions.append(occlusion)
        mlvl_vertexes = torch.cat(mlvl_vertexes)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_centerness = torch.cat(mlvl_centerness)
        mlvl_occlusions = torch.cat(mlvl_occlusions)
        for i in range(0, num_classes):
            cls_inds = mlvl_scores[:, i] > self.score_threshold[i]
            if not cls_inds.any():
                continue
            v = mlvl_vertexes[cls_inds, :][:,:self.joints[i]*2]
            scores = mlvl_scores[cls_inds, i]
            scores *= mlvl_centerness[cls_inds]
            occl = mlvl_occlusions[cls_inds, :]
            keep = self.nms_vertex(v, scores, 10)
            # keep = nms(self.poly_to_box(v), scores, self.nms_threshold)
            for idx in keep:
                polygon = v[idx]
                polygon[::2] = polygon[::2] / self.img_size[1]
                polygon[1::2] = polygon[1::2] / self.img_size[0]
                polygon = polygon.data.cpu().tolist()[:self.joints[i] * 2]
                score = scores[idx]
                occl_keep = '%s' * self.joints[i] % tuple(occl[idx].reshape(-1, 2).max(-1).indices.cpu().detach().numpy())[:self.joints[i]]
                poly = Shape(polygon, label=self.index_to_label[i], label_index=i, score=score, occlusions=occl_keep, relative=True)
                detected_shapes.append(poly)
        return detected_shapes

    def get_points(self, featmap_sizes, dtype, device):
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(self.get_points_single(featmap_sizes[i], self.strides[i], dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def nms_vertex(self, vertexes, scores, threshold=30):
        xc = vertexes[:,::2].sum(1) / vertexes[:,::2].size(1)
        yc = vertexes[:,1::2].sum(1) / vertexes[:,1::2].size(1)
        _, order = scores.sort(0, descending=True)
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()
                keep.append(i)
            xc_order = xc[order[1:]]
            yc_order = yc[order[1:]]
            distance = torch.sqrt((xc_order - xc[i]) * (xc_order - xc[i]) + (yc_order - yc[i]) * (yc_order - yc[i]))
            idx = (distance >= threshold).nonzero().squeeze()
            if idx.numel() == 0:
                break
            order = order[idx+1]
        return torch.LongTensor(keep)

    def poly_to_box(self, poly):
        xmin, _ = poly[:,::2].min(1)
        xmax, _ = poly[:,::2].max(1)
        ymin, _ = poly[:,1::2].min(1)
        ymax, _ = poly[:,1::2].max(1)
        return torch.stack([xmin, ymin, xmax, ymax], dim=1)
