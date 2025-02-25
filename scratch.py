#%%
import torch as t
import transformer_lens
from transformers import AutoModelForCausalLM


device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
print(device)
# %%
r1QwenDistillSmall = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
print(dir(r1QwenDistillSmall))

# %%

model = transformer_lens.HookedTransformer.from_pretrained_no_processing(
    "Qwen/Qwen2.5-7B",
    r1QwenDistillSmall,
    device=device
)

# %%
output = model.generate("There are three cities: The city of Gold, The city of Iron, and The City of Rubber. \nThree wise men" +
                        " approach the cities: One young, one tall, and one bold. \nWhich man goes to which city?<think>\n", max_new_tokens=1000, temperature =0.6)
print(output)


