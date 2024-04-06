import torch.nn as nn

def create_model(args, log):
    if args.arch == 'wideresnet':
        import models.wideresnet as models
        model = models.build_wideresnet(depth=args.model_depth,
                                        widen_factor=args.model_width,
                                        dropout=0,
                                        num_classes=args.num_classes)
    elif args.arch == 'resnext':
        import models.resnext as models
        model = models.build_resnext(cardinality=args.model_cardinality,
                                        depth=args.model_depth,
                                        width=args.model_width,
                                        num_classes=args.num_classes)
    log("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters())/1e6))
    return model