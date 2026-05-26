import re
import torch.nn as nn
from LinearFQ import LinearFQ
from ConvFQ import ConvFQ

def replace_modules(model, old_class=nn.Linear, new_class=LinearFQ, new_class_kwargs={}, match_name="", prefix=""):
    for name, module in model.named_children():
        qual_name = f"{prefix}.{name}" 
        print("qual_name module:", qual_name)

        if isinstance(module, old_class) and re.search(match_name, qual_name) is not None:
            kwargs = dict(new_class_kwargs)
            print("Matched module for replacement:", qual_name)

            if old_class == nn.Linear:
                print("Replacing Linear with LinearFQ:", qual_name)
                kwargs["in_features"] = module.in_features
                kwargs["out_features"] = module.out_features
                bias = getattr(module, "bias", None) is not None
                kwargs["bias"] = bias
                kwargs["device"] = module.weight.device
                new_module = new_class(**kwargs)
            elif old_class == nn.Conv2d:
                print("Replacing Conv2d with ConvFQ:", qual_name)
                kwargs["in_channels"] = module.in_channels
                kwargs["out_channels"] = module.out_channels
                kwargs["kernel_size"] = module.kernel_size
                kwargs["stride"] = module.stride
                kwargs["padding"] = module.padding
                bias = getattr(module, "bias", None) is not None
                kwargs["bias"] = bias
                kwargs["device"] = module.weight.device
                new_module = new_class(**kwargs)
            else:
                raise ValueError(f"Unsupported old_class: {old_class}")
            new_module.weight.data = module.weight.data
            if bias:
                new_module.bias.data = module.bias.data
            setattr(model, name, new_module)
        else:
            replace_modules(module, old_class, new_class, new_class_kwargs, match_name, prefix=qual_name)

def frankensteinize(model, old_class=nn.Linear, new_class=LinearFQ, new_class_kwargs={}):
    # Updated regex to include OLMo3 and other decoder-only models
    match_pattern = "conv|fc|cnn|linear|attention|classifier|T5Attention|ViTOutput|intermediate|.decoder.block|.encoder.block|olmo|feed_forward|output_projection|attention_proj|q_proj|k_proj|v_proj"
    replace_modules(model, old_class=old_class, new_class=new_class, new_class_kwargs=new_class_kwargs, match_name=match_pattern, prefix="")
    return model