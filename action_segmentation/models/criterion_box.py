import torch
import torch.nn.functional as F
from torch import nn

import torch.distributed as dist
from detectron2.utils.comm import get_world_size

# from detectron2.config import configurable
from ..config import configurable

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


def dice_loss(inputs, targets, num_masks):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)  ## same to IOU
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid() #input[num_mask, L] target[num_mask, L], so the num_mask is same to batchsize, and BCE on each mask
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets) ## p_t within 0~1
    loss = ce_loss * ((1 - p_t) ** gamma) ## 1-p_t within 0~1, so gamma >1?

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks

class SetCriterion_box(nn.Module):
    @configurable
    def __init__(self, num_classes, matcher, weight_dict, losses):
        super().__init__()
        self.num_classes = num_classes # for fill and compute empty_weight
        self.matcher = matcher
        self.weight_dict = weight_dict
        empty_weight = torch.ones(self.num_classes + 1)
        # empty_weight[-1] = self.eos_coef
        empty_weight[-1] = 0.01
        self.losses = losses

    @classmethod
    def from_config(cls, cfg):
        bbox_weight = cfg.model.bbox_weight
        giou_weight = cfg.model.giou_weight
        num_classes = cfg.dataset.n_classes

        matcher = HungarianMatcher(
            cost_class=1,
            cost_bbox=bbox_weight,
            cost_giou=giou_weight,
        )
        weight_dict = {"loss_ce": 1, "loss_bbox": bbox_weight, "loss_giou": giou_weight} # for combine all the loss
        losses = ['labels', 'boxes' ]
        return {
            'num_classes': num_classes,
            'matcher': matcher,
            'weight_dict': weight_dict,
            'losses': losses
        }


    def loss_labels(self, outputs, targets, indices, num_masks):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"] ##[bs, num_queries, cls] ?? each query has cls prob

        idx = self._get_src_permutation_idx(indices) #indices:tuple[bs](src_id(in order),tgt_id); idx:tuple batch_idx, src_idx (in order)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o

        # loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, label_smoothing=0.1)
        print(src_logits.argmax(-1))
        print(target_classes.sort().values)
        losses = {"loss_ce": loss_ce}
        return losses


    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_xl_to_xy(src_boxes),
            box_xl_to_xy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx] # selected src mask, same size as target
        masks = [t["masks"] for t in targets] # all target_mask [target_mask size]
        # TODO use valid to mask invalid areas due to padding in loss
        # target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = torch.stack(masks, dim=0)
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # src_masks = F.interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
        # src_masks = src_masks[:, 0].flatten(1)
        #
        # target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_masks), ##use the matched pair to get loss, which is also used for cost in processing of matching
            "loss_dice": dice_loss(src_masks, target_masks, num_masks),
        }
        return losses


    def _get_src_permutation_idx(self, indices):  ## find target according to the prediction
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])  ## full the tensor with same size as scr with batch_idx
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):  ## loss ："labels" or "masks"
        loss_map = {
            "labels": self.loss_labels,
            'boxes': self.loss_boxes,
            }  ## just find the loss functions here
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor( num_boxes, dtype=torch.float, device=next(iter(outputs.values())).device )

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()  ###? why clamp here

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))  # outputs targets [label, mask]

        # if "aux_outputs" in outputs:
        #     for i, aux_outputs in enumerate(outputs["aux_outputs"]):
        #         indices = self.matcher(aux_outputs, targets)
        #         for loss in self.losses:
        #             l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
        #             l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
        #             losses.update(l_dict)
        return losses

def box_area(boxes):
    x, y = boxes[:, 0], boxes[:, 1]
    return y-x

def box_xl_to_xy(x):
    s, l = x.unbind(-1)
    b = [s, (s + l)]
    return torch.stack(b, dim=-1)

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 1], boxes2[:, 1])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 1] >= boxes1[:, 0]).all()
    assert (boxes2[:, 1] >= boxes2[:, 0]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    rb = torch.max(boxes1[:, None, 1], boxes2[:, 1])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh

    return iou
    # return iou - (area - union) / area


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def batch_dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid() ##[num_queries, HW]
    inputs = inputs.flatten(1) ##[nun_queries, HW]
    # numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets) ##targets[num_total_objects, HW]
    numerator = 2 * torch.matmul(inputs, targets.transpose(-1,-2)) ##targets[num_total_objects, HW]

    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1) ##more overlap, less loss, so 1-IOU
    return loss


def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1] ## input [num_queries, HW]

    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none" # in 1/N, the prob of each frame belongs to 1, N same as the bachsize
    )
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    # loss = torch.einsum("nc,mc->nm", focal_pos, targets) + torch.einsum(
    #     "nc,mc->nm", focal_neg, (1 - targets)   #mix all the results here according to the targets
    # )  ##get nm like similarity metrix for pos and neg

    loss = torch.matmul(focal_pos, targets.transpose(-1,-2)) + torch.matmul( focal_neg, (1 - targets).transpose(-1,-2))
    return loss / hw


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_xl_to_xy(out_bbox), box_xl_to_xy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # indices = [(list(range(sizes[i])), list(range(sizes[i]))) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


