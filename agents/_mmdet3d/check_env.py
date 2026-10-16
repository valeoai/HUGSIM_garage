"""Checks that the compiled ops of this environment run on the GPU and agree with their references.

    CUDA_VISIBLE_DEVICES=0 pixi run --manifest-path agents/_mmdet3d/pixi.toml python agents/_mmdet3d/check_env.py

A torch build without kernels for the card does not fail on a matmul, it returns zeros, so this
compares against known values and against the pure-PyTorch / CPU paths of each op.
"""

import sys

import torch


def check(name, ok, detail=''):
    print(f'{"PASS" if ok else "FAIL"}  {name}  {detail}', flush=True)
    return ok


def main():
    results = []
    print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0),
          'capability', torch.cuda.get_device_capability(0))
    torch.manual_seed(0)

    a = torch.randn(256, 256, device='cuda')
    b = torch.randn(256, 256, device='cuda')
    err = (a @ b - (a.cpu().double() @ b.cpu().double()).float().cuda()).abs().max().item()
    results.append(check('matmul vs CPU float64', a.abs().sum().item() > 0 and err < 1e-2, f'max_err={err:.2e}'))

    import mmcv
    from mmcv.ops import ModulatedDeformConv2d, nms_rotated  # noqa: F401
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch

    print('mmcv', mmcv.__version__)
    bs, nq, heads, ch, levels, points = 2, 50, 8, 32, 4, 4
    shapes = torch.tensor([[40, 40], [20, 20], [10, 10], [5, 5]], device='cuda')
    starts = torch.cat([shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]])
    n_value = int(shapes.prod(1).sum())
    value = torch.randn(bs, n_value, heads, ch, device='cuda')
    loc = torch.rand(bs, nq, heads, levels, points, 2, device='cuda')
    weight = torch.rand(bs, nq, heads, levels, points, device='cuda')
    weight = weight / weight.sum((-1, -2), keepdim=True)
    out_cuda = MultiScaleDeformableAttnFunction.apply(value, shapes, starts, loc, weight, 64)
    out_ref = multi_scale_deformable_attn_pytorch(value, shapes, loc, weight)
    err = (out_cuda - out_ref).abs().max().item()
    results.append(check('MultiScaleDeformableAttn CUDA vs pytorch', out_ref.abs().max().item() > 0 and err < 1e-4,
                         f'max_err={err:.2e}'))

    conv = ModulatedDeformConv2d(16, 16, 3, padding=1, deform_groups=2).cuda()
    x = torch.randn(2, 16, 24, 24, device='cuda')
    offset = torch.randn(2, 2 * 2 * 9, 24, 24, device='cuda')
    mask = torch.rand(2, 2 * 9, 24, 24, device='cuda')
    y_cuda = conv(x, offset, mask)
    conv_cpu = ModulatedDeformConv2d(16, 16, 3, padding=1, deform_groups=2)
    conv_cpu.load_state_dict({k: v.cpu() for k, v in conv.state_dict().items()})
    y_cpu = conv_cpu(x.cpu(), offset.cpu(), mask.cpu())
    err = (y_cuda.cpu() - y_cpu).abs().max().item()
    results.append(check('ModulatedDeformConv2d CUDA vs CPU', y_cpu.abs().max().item() > 0 and err < 1e-3,
                         f'max_err={err:.2e}'))

    import mmdet3d
    from mmdet3d.ops import points_in_boxes_all

    print('mmdet3d', mmdet3d.__version__)
    pts = torch.tensor([[[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]], device='cuda')
    boxes = torch.tensor([[[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]]], device='cuda')
    inside = points_in_boxes_all(pts, boxes)[0, :, 0].tolist()
    results.append(check('points_in_boxes_all (mmdet3d lidar box ops)', inside == [1, 0], str(inside)))

    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
