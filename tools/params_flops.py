import sys
sys.path.append('.')
from train import *
from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count, parameter_count
from rscd.models.backbones.quad_util.csms6s import flops_selective_scan_fn, flops_selective_scan_ref, selective_scan_flop_jit

def parse_args():
    parser = argparse.ArgumentParser(description='count params and flops')
    parser.add_argument("-c", "--config", type=str, default="configs/cdlama.py")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    return args

def flops_mamba(model, shape=(3, 224, 224)):
    # shape = self.__input_shape__[1:]
    supported_ops = {
        "aten::silu": None,  # as relu is in _IGNORED_OPS
        "aten::neg": None,  # as relu is in _IGNORED_OPS
        "aten::exp": None,  # as relu is in _IGNORED_OPS
        "aten::flip": None,  # as permute is in _IGNORED_OPS
        # "prim::PythonOp.CrossScan": None,
        # "prim::PythonOp.CrossMerge": None,
        "prim::PythonOp.SelectiveScanCuda": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit,
    }

    model.cuda().eval()

    input1 = torch.randn((1, *shape), device=next(model.parameters()).device)
    input2 = torch.randn((1, *shape), device=next(model.parameters()).device)
    params = parameter_count(model)[""]
    Gflops, unsupported = flop_count(model=model, inputs=(input1,input2), supported_ops=supported_ops)

    del model, input1, input2
    # return sum(Gflops.values()) * 1e9
    return f"params {params / 1e6} GFLOPs {sum(Gflops.values())}"

if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    net = myTrain(cfg).net.cuda()

    size = args.size
    input = torch.rand((1, 3, size, size)).cuda()
    
    net.eval()
    flops = FlopCountAnalysis(net, (input, input))
    print(flop_count_table(flops, max_depth = 2))

    print(flops_mamba(net, (3, size, size)))
