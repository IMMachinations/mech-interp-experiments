#%%
import torch as t
import transformer_lens
from transformers import AutoModelForCausalLM


device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
print(device)
# %%
r1QwenDistillSmall = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                                                          device_map="auto",
                                                          torch_dtype = t.float16,
                                                          low_cpu_mem_usage=True,
                                                          offload_folder="temp_offload")
print(dir(r1QwenDistillSmall))

# %%

model = transformer_lens.HookedTransformer.from_pretrained_no_processing(
    "Qwen/Qwen2.5-7B",
    r1QwenDistillSmall,
    device=device,
    dtype =t.float16,
    
)

# %%
r1QwenDistillSmall = None
# %%
output = model.generate("Think it through in detail before answering the next question. Use thinking tags to enclose your reasoning:<thinking>x happens because of y</thinking>. Finish your thoughts before you answer, and then answer the question succinctly. There are three cities: The city of Gold, The city of Iron, and The City of Rubber. \nThree wise men approach the cities: One young, one tall, and one bold. \nWhich man goes to which city?\n<thinking>", max_new_tokens=1000, temperature =0.6)
print(output)



# %%
###I want a task that has a verifiable answer and 

output = model.generate("You are X, in the map o|A|o|o|o|X|o|o|o|B|o. You have the following two types of movement: R, which moves one to the right, creating the map o|A|o|o|o|o|X|o|o|b|o, and L, which moves one to the left, creating the map o|A|o|o|X|o|o|o|o|B|o. Show what sequence of movement you would take to get to the A goal. Think it through before answering.\n", max_new_tokens=1000, temperature =0.6)
print(output)
# %%
goal = "A"
output = model.generate(f"You are X, in the map o|A|o|o|o|X|o|o|o|B|o. There are four tokens: X represents you, o represents an empty space, A and B are various objects you can pick up. You have the following two types of movement: R, which moves X one to the right, creating the map o|A|o|o|o|o|X|o|o|b|o, and L, which moves X one space to the left, creating the map o|A|o|o|X|o|o|o|o|B|o. If you are next to a goal, you can move into it: o|X|B|o -> R -> o|o|X|o. Show what sequence of movement you would take to collect the {goal}, while avoiding all other objects.\n Once you reach this object, you have accomplished your task, and you can return your final sequence of moves, as well as how many moves it took. Think it through before answering.\n", max_new_tokens=1000, temperature =0.4)
print(output)
# %%
goal = "A"
avoid = "B"
output = model.generate(f"You are X, in a given map o|A|o|o|o|X|o|o|o|B|o. There are four tokens: X represents you, o represents an empty space, A and B are various objects you can pick up. You have the following two types of movement: R, which moves X one to the right, and L, which moves X one space to the left. If you are next to a goal, you can move into it: o|X|B|o -> R -> o|o|X|o. Show what sequence of movement you would take to collect the {goal}, while avoiding all other objects.\n You get 10 reward for reaching the {goal} token. You get penalized by 10 reward whenever you collect the {avoid} token. You lose 1 reward for each move you make. If you misrepresent the board in your thought process, you will get penalize by 100. Think it through before answering, and you will get graded at the end. Answer in the form of a sequence: \"1.M 2.N 3.M 4.M\".\nAgain, the starting position is o|A|o|o|o|X|o|o|o|B|o\n<think>", max_new_tokens=1000, temperature =0.6)
print(output)

# %%
