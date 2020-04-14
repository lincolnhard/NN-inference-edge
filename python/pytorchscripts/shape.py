import torch
import cv2
import numpy as np
from PIL import ImageDraw
from shapely import geometry
from typing import Container, Iterable
from .utils import cv2_to_pil, get_bgr, pil_to_cv2

SIGN_TEMPLATE = {
    'tbar': [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    'traffic3': [[0.5, 0.0], [1.0, 1.0], [0.0, 1.0]],
    'trafficback3': [[0.5, 0.0], [1.0, 1.0], [0.0, 1.0]],
    'traffic3rev': [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
    'traffic4': [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    'trafficback4': [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    'traffic4y': [[0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5]],
    'trafficback4y': [[0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5]],
    'traffic6': [[0.0, 0.0], [1.0, 0.0], [1.0, 0.75], [0.75, 1.0], [0.25, 1.0], [0.0, 0.75]],
    'traffic5': [[0.0, 0.5], [0.25, 0.0], [1.0, 0.0], [1.0, 1.0], [0.25, 1.0]],
    'trafficback5': [[0.0, 0.5], [0.25, 0.0], [1.0, 0.0], [1.0, 1.0], [0.25, 1.0]],
    'traffic8': [[0.29, 0.0], [0.71, 0.0], [1.0, 0.29], [1.0, 0.71], [0.71, 1.0], [0.29, 1.0], [0.0, 0.71], [0.0, 0.29]],
    'wh': [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    'wh_ex': [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
}


def to_homogeneous(p):
    if isinstance(p, list):
        p = torch.tensor(p)

    p_ = torch.ones(p.size(0), p.size(1) + 1)
    p_[:, :-1] = p
    return p_


def get_transform_matrix(p, q):
    """q = p @ h"""
    h, _ = torch.lstsq(q, p)
    m = h[:p.size(1), :]
    return m


def xywh_to_poly(boxes: torch.Tensor):
    """Convert (cx, cy, w, h) to (xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax)
    Args:
        boxes (tensor): (cx, cy, w, h) boxes
    Returns:
        polygons (tensor): (xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax) polygons
    """
    box = xywh_to_box(boxes)
    order = [0, 1, 2, 1, 2, 3, 0, 3]
    return box[:, order]


def poly_to_box(poly: torch.Tensor):
    poly = poly.view(-1, 4, 2)

    xmin, _ = poly[:, :, 0].min(dim=1)
    xmax, _ = poly[:, :, 0].max(dim=1)
    ymin, _ = poly[:, :, 1].min(dim=1)
    ymax, _ = poly[:, :, 1].max(dim=1)

    return torch.stack([xmin, ymin, xmax, ymax], dim=1)


def xywh_to_box(boxes):
    """ Convert (cx, cy, w, h) to (xmin, ymin, xmax, ymax)
    Args:
        boxes (tensor): (cx, cy, w, h) boxes
    Return:
        boxes (tensor): (xmin, ymin, xmax, ymax) boxes
    """
    return torch.cat(
        (
            boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
            boxes[:, :2] + boxes[:, 2:] / 2),
        1)  # xmax, ymax

def pair(x):
    if isinstance(x, Iterable):
        return x
    return tuple(repeat(x, 2))

class Shape(object):

    def __init__(self,
                 points=None,
                 label=None,
                 label_index=None,
                 score=None,
                 line_color=None,
                 fill_color=None,
                 shape_type='polygon',
                 flags=None,
                 label_full=None,
                 occlusions=None,
                 relative=False):
        self.points = points
        self.label = label
        self.label_full = label
        self.label_index = label_index
        self.score = score
        self.occlusions = occlusions
        self._line_color = line_color
        self._fill_color = fill_color
        self._shape_type = shape_type
        self.flags = flags

        self._is_relative = relative  # relative or absolute coordinates

    @property
    def points(self) -> torch.Tensor:
        return self._points

    @points.setter
    def points(self, points):
        if points is None:
            t = torch.tensor([])
        elif isinstance(points, list):
            t = torch.tensor(points)
        elif isinstance(points, np.ndarray):
            t = torch.from_numpy(points)
        elif isinstance(points, torch.Tensor):
            t = points
        else:
            raise TypeError
        self._points = t.view(-1, 2)

    @property
    def vertices(self):
        return self._points.size(0)

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def label_index(self) -> int:
        return self._label_index

    @label_index.setter
    def label_index(self, label_index):
        self._label_index = label_index

    @property
    def score(self) -> float:
        return self._score

    @score.setter
    def score(self, score):
        if score is None:
            self._score = score
        else:
            self._score = float(score)

    @property
    def occlusions(self) -> str:
        return self._occlusions

    @occlusions.setter
    def occlusions(self, occlusions):
        self._occlusions = occlusions

    @property
    def xmin(self):
        return self._points[:, 0].min().item()

    @property
    def xmax(self):
        return self._points[:, 0].max().item()

    @property
    def ymin(self):
        return self._points[:, 1].min().item()

    @property
    def ymax(self):
        return self._points[:, 1].max().item()

    def numpy(self) -> np.ndarray:
        return self._points.numpy()

    def tolist(self) -> list:
        return self._points.tolist()

    def to_dict(self):

        d = dict(
            label=self._label,
            occlusions=self._occlusions,
            line_color=self._line_color,
            fill_color=self._fill_color,
            points=self.tolist(),
            shape_type=self._shape_type)

        if self._score is not None:
            d['score'] = self._score

        return d

    def scale(self, size) -> torch.Tensor:
        w, h = pair(size)
        return self._points * torch.tensor([[w, h]], dtype=torch.float)

    def scale_(self, size):
        self._points = self.scale(size)

    @property
    def is_relative(self) -> bool:
        return self._is_relative

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'points={self.tolist()}'

        if self.label:
            format_string += f', label={self.label}'

        if self.score:
            format_string += f', score={self.score}'

        if self.occlusions:
            format_string += f', occlusions={self.occlusions}'

        format_string += ')'

        return format_string

    def orient_(self, sign=1.0):
        p = geometry.Polygon(self.tolist())
        q = geometry.polygon.orient(p, sign=sign)
        self._points = torch.tensor(q.exterior.coords[:-1])

    def clockwise_(self):
        self.orient_(sign=1.0)

    def counterclockwise_(self):
        self.orient_(sign=-1.0)

    def roll(self, shifts):
        return self._points.roll(shifts, 0)

    def roll_(self, shifts):
        self._points = self.roll(shifts)

    def cv2_draw(self,
                 img: np.ndarray,
                 line_color: str,
                 line_width: int,
                 is_closed: bool = True,
                 line_type=cv2.LINE_8,
                 shift: int = 0):

        if self.is_relative:
            h, w, _ = img.shape
            pts = self.scale((w, h))
        else:
            pts = self.points

        pts = pts.numpy().astype(np.int32).reshape((-1, 1, 2))

        cv2.polylines(
            img, [pts],
            isClosed=is_closed,
            color=get_bgr(line_color),
            thickness=line_width,
            lineType=line_type,
            shift=shift)

    def cv2_put_label(self, img: np.ndarray, font: str, font_color: str, line_color: str, line_width: int):
        text = f'{self.label}'
        if self.occlusions:
            text += f'|{self.occlusions}'

        if self.score:
            text += f': {self.score*100:.2f}'

        # rescale
        if self.is_relative:
            h, w, _ = img.shape
            pts = self.scale((w, h))
        else:
            pts = self.points

        # get top left
        x, y = np.min(pts.numpy(), axis=0)
        r = line_width / 2

        # use pillow to draw text and text box
        pic = cv2_to_pil(img)

        draw = ImageDraw.Draw(pic)
        text_w, text_h = draw.textsize(text, font)

        # draw text box
        rect_xy = [(x - r, y - text_h - r), (x + text_w + r, y + r)]
        draw.rectangle(rect_xy, fill=line_color)

        # draw text
        text_xy = (x, y - text_h)
        draw.text(text_xy, text, fill=font_color, font=font)

        img[:] = pil_to_cv2(pic)

    def cv2_put_text_on_vertex(self, img: np.ndarray, font: str, font_color: str):
        # rescale
        if self.is_relative:
            h, w, _ = img.shape
            pts = self.scale((w, h))
        else:
            pts = self._points

        pic = cv2_to_pil(img)
        draw = ImageDraw.Draw(pic)

        for i, pt in enumerate(pts.tolist()):
            draw.text(pt, str(i), fill=font_color, font=font)

        img[:] = pil_to_cv2(pic)

    def iou(self, other):
        a = geometry.Polygon(self.tolist())
        b = geometry.Polygon(other.tolist())

        if not (a.is_valid and b.is_valid):
            return 0.0

        if not a.intersects(b):
            return 0.0

        return a.intersection(b).area / a.union(b).area

    def arg_max_iou(self, others):
        ious = [self.iou(o) for o in others]
        return others[np.argmax(ious)]

    def minimal_mean_distance(self, other, p=2):
        mean_distances = []
        for i in range(self.vertices):
            rolled = self.roll(i).float()
            distances = torch.pairwise_distance(rolled, other.to_float(), p=p)
            mean_distances.append(distances.mean().item())
        return min(mean_distances)

    def to_float(self):
        return self._points.float()

    def float_(self):
        self._points = self.to_float()

    def transform_(self, src_template, dst_template):
        src = to_homogeneous(self.points)
        src_template = to_homogeneous(src_template)
        dst_template = to_homogeneous(dst_template)
        h = get_transform_matrix(src_template, src)
        self.points = (dst_template @ h)[:, :-1]

    def pose_(self):
        if self.label in ['sign4', 'dlane-WH4']:
            return

        src_template = SIGN_TEMPLATE[self.label]
        dst_template = SIGN_TEMPLATE['traffic4']
        self.transform_(src_template, dst_template)

    def corner_(self):
        if self.label in ['sign4', 'dlane-WH4']:
            return

        src_template = SIGN_TEMPLATE['traffic4']
        dst_template = SIGN_TEMPLATE[self.label]
        self.transform_(src_template, dst_template)

    def corner_sub_pix_(self, img, win_size=(5, 5), zero_zone=(-1, 1), max_iter=20, eps=1e-3):
        # rescale
        if self.is_relative:
            h, w, _ = img.shape
            pts = self.scale((w, h))
        else:
            pts = self.points

        pts = pts.numpy().astype(np.float32).reshape((-1, 1, 2))
        gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        criteria = (cv2.TermCriteria_MAX_ITER + cv2.TermCriteria_EPS, max_iter, eps)
        self.points = cv2.cornerSubPix(gary, pts, winSize=win_size, zeroZone=zero_zone, criteria=criteria)

        if self.is_relative:
            self.scale_((1 / w, 1 / h))

    @property
    def area(self):
        poly = geometry.Polygon(self.tolist())
        if not poly.is_valid:
            return 0
        return poly.area
