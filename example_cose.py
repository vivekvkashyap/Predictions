import jax
print(jax.local_device_count())
import jax.numpy as jnp

import flax
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict, unfreeze
from flax.training.common_utils import get_metrics,onehot,shard,shard_prng_key

from transformers import  GPTNeoConfig
from transformers.models.gpt_neo.modeling_flax_gpt_neo import FlaxGPTNeoPreTrainedModel
from transformers import GPT2Tokenizer

from datasets import load_dataset,Dataset
import pandas as pd

num_choices=5

cos0={'question':'A desert is a large area of dry land where is little amount of water, but the globe is mostly covered by an what?','choices':['ocean', 'australia', 'asia', 'continent', 'island']}
cos1={'question':'If one wants to breathe while they sleep, air needs to be inside of what?','choices': ['hotel','surface of earth','space shuttle','supermarket balloon','home']}
cos2={'question':'If you want to kill someone you can do what to them with a knife?','choices':['sip through','damnation','shoot','commit crime','peirce through their heart']}
cos3={'question':'By learning about the globe, many poor college students gain what?','choices':['pleasure','greater mobility','desire to travel','global warming','time to take a break']}
cos4={'question':'A man takes a seat while others are watching a the big screen, where is he likely?','choices': ['in cinema', 'martorell', 'falling down', 'show', 'airplane']}
cos5={'question':'He needed more information to fix it, so he consulted the what?','choices': ['chickens', 'google', 'newspaper', 'online', 'manual']}
cos_total=[cos0,cos1,cos2,cos3,cos4,cos5]
cos={}
cos['question']=[i['question'] for i in cos_total]
cos['choices']=[i['choices'] for i in cos_total]
cos_dataset = Dataset.from_dict(cos)


def preprocess(example):
    example['first_sentence']=[example['question']]*num_choices
    example['second_sentence']=[example['choices'][i] for i in range(num_choices)]
    example['label']=example['choices'].index(example['answer'])
    return example

ccos_dataset=cos_dataset.map(preprocess)

len_test_dataset=4

tokenizer=GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B',pad_token='<|endoftext|>')

remove_col=ccos_dataset.column_names

def tokenize(examples):
    tokenized_examples=tokenizer(examples['first_sentence'],examples['second_sentence'],padding='max_length',truncation=True,max_length=256,return_tensors='jax')
    return tokenized_examples

test_dataset=ccos_dataset.map(tokenize)

test_dataset=test_dataset.remove_columns(remove_col)
list1=[]

def glue_test_data_loader(rng,dataset,batch_size):
  steps_per_epoch=len_test_dataset//batch_size
  perms=jax.random.permutation(rng,len_test_dataset)
  perms=perms[:steps_per_epoch*batch_size]
  perms=perms.reshape((steps_per_epoch,batch_size))
  for perm in perms:
    list1.append(perm)
    batch=dataset[perm]
    #print(jnp.array(batch['label']))
    batch={k:jnp.array(v) for k,v in batch.items()}
    #batch=shard(batch)
    yield batch

seed=0
rng=jax.random.PRNGKey(seed)
dropout_rngs=jax.random.split(rng,jax.local_device_count())

input_id=jnp.array(test_dataset['input_ids'])
att_mask=jnp.array(test_dataset['attention_mask'])

total_batch_size=1

from  model_file  import FlaxGPTNeoForMultipleChoice

model = FlaxGPTNeoForMultipleChoice.from_pretrained('Vivek/gptneo_cose',input_shape=(1,num_choices,1))

restored_output=[]
rng, input_rng = jax.random.split(rng)
b=jnp.array(test_dataset['input_ids'])
c=jnp.array(test_dataset['attention_mask'])
outputs=model(b,c)
print(outputs)
final_output=jnp.argmax(outputs,axis=-1)
restored_output.append(final_output)

finall=pd.DataFrame({'predictions':restored_output})
finall.to_csv('./robustness_cose.csv')