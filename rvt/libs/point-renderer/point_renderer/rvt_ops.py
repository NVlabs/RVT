from typing import Optional, Tuple
import torch
import math


# source: https://discuss.pytorch.org/t/batched-index-select/9115/6
def batched_index_select(inp, dim, index):
    """
    input: B x * x ... x *
    dim: 0 < scalar
    index: B x M
    """
    views = [inp.shape[0]] + [1 if i != dim else -1 for i in range(1, len(inp.shape))]
    expanse = list(inp.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(inp, dim, index)


# TODO: break into two functions
def select_feat_from_hm(
    pt_cam: torch.Tensor, hm: torch.Tensor, pt_cam_wei: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor]:
    """
    :param pt_cam:
        continuous location of point coordinates from where value needs to be
        selected. it is of size [nc, npt, 2], locations in pytorch3d screen
        notations
    :param hm: size [nc, nw, h, w]
    :param pt_cam_wei:
        some predifined weight of size [nc, npt], it is used along with the
        distance weights
    :return:
        tuple with the first element being the wighted average for each point
        according to the hm values. the size is [nc, npt, nw]. the second and
        third elements are intermediate values to be used while chaching
    """
    nc, nw, h, w = hm.shape
    npt = pt_cam.shape[1]
    if pt_cam_wei is None:
        pt_cam_wei = torch.ones([nc, npt]).to(hm.device)

    # giving points outside the image zero weight
    pt_cam_wei[pt_cam[:, :, 0] < 0] = 0
    pt_cam_wei[pt_cam[:, :, 1] < 0] = 0
    pt_cam_wei[pt_cam[:, :, 0] > (w - 1)] = 0
    pt_cam_wei[pt_cam[:, :, 1] > (h - 1)] = 0

    pt_cam = pt_cam.unsqueeze(2).repeat([1, 1, 4, 1])
    # later used for calculating weight
    pt_cam_con = pt_cam.detach().clone()

    # getting discrete grid location of pts in the camera image space
    pt_cam[:, :, 0, 0] = torch.floor(pt_cam[:, :, 0, 0])
    pt_cam[:, :, 0, 1] = torch.floor(pt_cam[:, :, 0, 1])
    pt_cam[:, :, 1, 0] = torch.floor(pt_cam[:, :, 1, 0])
    pt_cam[:, :, 1, 1] = torch.ceil(pt_cam[:, :, 1, 1])
    pt_cam[:, :, 2, 0] = torch.ceil(pt_cam[:, :, 2, 0])
    pt_cam[:, :, 2, 1] = torch.floor(pt_cam[:, :, 2, 1])
    pt_cam[:, :, 3, 0] = torch.ceil(pt_cam[:, :, 3, 0])
    pt_cam[:, :, 3, 1] = torch.ceil(pt_cam[:, :, 3, 1])
    pt_cam = pt_cam.long()  # [nc, npt, 4, 2]
    # since we are taking modulo, points at the edge, i,e at h or w will be
    # mapped to 0. this will make their distance from the continous location
    # large and hence they won't matter. therefore we don't need an explicit
    # step to remove such points
    pt_cam[:, :, :, 0] = torch.fmod(pt_cam[:, :, :, 0], int(w))
    pt_cam[:, :, :, 1] = torch.fmod(pt_cam[:, :, :, 1], int(h))
    pt_cam[pt_cam < 0] = 0

    # getting normalized weight for each discrete location for pt
    # weight based on distance of point from the discrete location
    # [nc, npt, 4]
    pt_cam_dis = 1 / (torch.sqrt(torch.sum((pt_cam_con - pt_cam) ** 2, dim=-1)) + 1e-10)
    pt_cam_wei = pt_cam_wei.unsqueeze(-1) * pt_cam_dis
    _pt_cam_wei = torch.sum(pt_cam_wei, dim=-1, keepdim=True)
    _pt_cam_wei[_pt_cam_wei == 0.0] = 1
    # cached pt_cam_wei in select_feat_from_hm_cache
    pt_cam_wei = pt_cam_wei / _pt_cam_wei  # [nc, npt, 4]

    # transforming indices from 2D to 1D to use pytorch gather
    hm = hm.permute(0, 2, 3, 1).view(nc, h * w, nw)  # [nc, h * w, nw]
    pt_cam = pt_cam.view(nc, 4 * npt, 2)  # [nc, 4 * npt, 2]
    # cached pt_cam in select_feat_from_hm_cache
    pt_cam = (pt_cam[:, :, 1] * w) + pt_cam[:, :, 0]  # [nc, 4 * npt]
    # [nc, 4 * npt, nw]
    pt_cam_val = batched_index_select(hm, dim=1, index=pt_cam)
    # tranforming back each discrete location of point
    pt_cam_val = pt_cam_val.view(nc, npt, 4, nw)
    # summing weighted contribution of each discrete location of a point
    # [nc, npt, nw]
    pt_cam_val = torch.sum(pt_cam_val * pt_cam_wei.unsqueeze(-1), dim=2)
    return pt_cam_val, pt_cam, pt_cam_wei


def select_feat_from_hm_cache(
    pt_cam: torch.Tensor,
    hm: torch.Tensor,
    pt_cam_wei: torch.Tensor,
) -> torch.Tensor:
    """
    Cached version of select_feat_from_hm where we feed in directly the
    intermediate value of pt_cam and pt_cam_wei. Look into the original
    function to get the meaning of these values and return type. It could be
    used while inference if the location of the points remain the same.
    """

    nc, nw, h, w = hm.shape
    # transforming indices from 2D to 1D to use pytorch gather
    hm = hm.permute(0, 2, 3, 1).view(nc, h * w, nw)  # [nc, h * w, nw]
    # [nc, 4 * npt, nw]
    pt_cam_val = batched_index_select(hm, dim=1, index=pt_cam)
    # tranforming back each discrete location of point
    pt_cam_val = pt_cam_val.view(nc, -1, 4, nw)
    # summing weighted contribution of each discrete location of a point
    # [nc, npt, nw]
    pt_cam_val = torch.sum(pt_cam_val * pt_cam_wei.unsqueeze(-1), dim=2)
    return pt_cam_val


# unit tests to verify select_feat_from_hm
def test_select_feat_from_hm():
    def get_out(pt_cam, hm):
        nc, nw, d = pt_cam.shape
        nc2, c, h, w = hm.shape
        assert nc == nc2
        assert d == 2
        out = torch.zeros((nc, nw, c))
        for i in range(nc):
            for j in range(nw):
                wx, hx = pt_cam[i, j]
                if (wx < 0) or (hx < 0) or (wx > (w - 1)) or (hx > (h - 1)):
                    out[i, j, :] = 0
                else:
                    coords = (
                        (math.floor(wx), math.floor(hx)),
                        (math.floor(wx), math.ceil(hx)),
                        (math.ceil(wx), math.floor(hx)),
                        (math.ceil(wx), math.ceil(hx)),
                    )
                    vals = []
                    total = 0
                    for x, y in coords:
                        val = 1 / (math.sqrt(((wx - x) ** 2) + ((hx - y) ** 2)) + 1e-10)
                        vals.append(val)
                        total += val

                    vals = [x / total for x in vals]

                    for (x, y), val in zip(coords, vals):
                        out[i, j] += val * hm[i, :, y, x]
        return out

    pt_cam_1 = torch.tensor([[[11.11, 120.1], [37.8, 0.0], [99, 76.5]]])
    hm_1_1 = torch.ones((1, 1, 100, 120))
    hm_1_2 = torch.ones((1, 1, 120, 100))
    out_1 = torch.ones((1, 3, 1))
    out_1[0, 0, 0] = 0

    pt_cam_2 = torch.tensor(
        [
            [[11.11, 12.11], [37.8, 0.0]],
            [[61.00, 12.00], [123.99, 123.0]],
        ]
    )
    hm_2_1 = torch.rand((2, 1, 200, 100))
    hm_2_2 = torch.rand((2, 1, 100, 200))

    test_sets = [
        (pt_cam_1, hm_1_1, out_1),
        (pt_cam_1, hm_1_2, out_1),
        (pt_cam_2, hm_2_1, get_out(pt_cam_2, hm_2_1)),
        (pt_cam_2, hm_2_2, get_out(pt_cam_2, hm_2_2)),
    ]

    for i, test in enumerate(test_sets):
        pt_cam, hm, out = test
        _out, _, _ = select_feat_from_hm(pt_cam, hm)
        out = out.float()
        if torch.all(torch.abs(_out - out) < 1e-5):
            print(f"Passed test {i}, {out}, {_out}")
        else:
            print(f"Failed test {i}, {out}, {_out}")
