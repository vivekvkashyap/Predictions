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

num_choices=4

cosmos0={'context':"I ' m back home in 20 minutes , just in time to make a couple of sandwiches for Teenager No . 2 who is out the door within ten minutes . Then I have forty minutes of me time before I have to get to work .",'question':'What will I do after I make the burger but before I go to work ?','answer0': 'None of the above choices .','answer1': 'I will take care of the kids .','answer2': 'I will clean up the house .','answer3': 'I will take a nap .'}
cosmos1={'context':'Well last Friday when we had the quiz , he was about to collect them , and I realized I had an addition mistake in my Gauss - Jordan , which was why my checking of the matrices did n\'t come out correctly . I asked him if he could pick up mine last ( I was the first one that he picks up ) because I found a simple mistake . He just told me to put " mistake " there and he will overlook it . Well he did n\'t .','question':'Why was someone about to collect the quizzes ?','answer0': 'None of the above choices .','answer1': 'Because the quiz was over .','answer2': 'Because I realized I had an addition mistake .','answer3': "Because my checking of the matrices did n't come out correctly ."}
cosmos2={'context':'I thought about robbing butter . Planned what I would wear , what types of butter to rob , what to drive . I just felt completely free of any sense of constrain . Those ideas faded to be replaced by other short - lived intense obsessions .','question': 'What did I consider stealing ?','answer0': 'Clothes','answer1': 'None of the above choices .','answer2': 'Cars','answer3': 'Money'}
cosmos3={'context':'I did glare at her , though . I was probably only in there 40 hours total . I knew what to say to get out , and I played their stupid little games . I swallowed their pills , and I agreed to enroll in their partial hospitalization program if they would just let me go home .','question': 'What did this person have to be where she was ?','answer0': 'Unemployment','answer1': 'Psychiatric issues','answer2': 'Homelessness','answer3': 'None of the above choices .'}
cosmos4={'context':'Several years ago , I was fortunate enough to review the SideSwipe blade for the mixer . The thing was a powerhouse , keeping my bowl sides scrapped and everything fully mixed every time . Unfortunately , I loved it so much that I wore it out .','question': 'How did your loving the blade cause it to wear out ?','answer0': "I was too afraid to use and damage it that it lost it 's sharpness over time",'answer1': 'I never used it because I did not want to damage it that it became useless','answer2': 'None of the above choices .','answer3': "I was unable to use it because I did n't want to wear it , when i went to use it it was dull"}
cosmos5={'context':"It was n't a good day today here in the Chicagoland . It was raining all day . I am supposed to go to the market store but I did n't feel like it . I just slept all day :D I admit it was n't good outside but it feels good to go to sleep when it 's raining .",'question': "What might be different if it was a good day ?",'answer0': 'They would have slept all day','answer1': 'They would have gone to the market store','answer2': "It would n't have been a good day",'answer3': "They would n't have gone to the market store"}
cosmos_total=[cosmos0,cosmos1,cosmos2,cosmos3,cosmos4,cosmos5]
cosmos={}
cosmos['context']=[i['context'] for i in cosmos_total]
cosmos['question']=[i['question'] for i in cosmos_total]
cosmos['answer0']=[i['answer0'] for i in cosmos_total]
cosmos['answer1']=[i['answer1'] for i in cosmos_total]
cosmos['answer2']=[i['answer2'] for i in cosmos_total]
cosmos['answer3']=[i['answer3'] for i in cosmos_total]
cosmos_dataset = Dataset.from_dict(cosmos)

def preprocess(example):
    example['context&question']=example['context']+example['question']
    example['first_sentence']=[example['context&question']]*num_choices
    example['second_sentence']=[example[f'answer{i}'] for i in range(num_choices)]
    return example

ccosmos_dataset=cosmos_dataset.map(preprocess)

len_test_dataset=4

tokenizer=GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B',pad_token='<|endoftext|>')

remove_col=ccosmos_dataset.column_names

def tokenize(examples):
    tokenized_examples=tokenizer(examples['first_sentence'],examples['second_sentence'],padding='max_length',truncation=True,max_length=256,return_tensors='jax')
    return tokenized_examples

test_dataset=ccosmos_dataset.map(tokenize)

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

model = FlaxGPTNeoForMultipleChoice.from_pretrained('Vivek/gptneo_cosmos',input_shape=(1,num_choices,1))

restored_output=[]
rng, input_rng = jax.random.split(rng)
b=jnp.array(test_dataset['input_ids'])
c=jnp.array(test_dataset['attention_mask'])
outputs=model(b,c)
print(outputs)
final_output=jnp.argmax(outputs,axis=-1)
restored_output.append(final_output)

finall=pd.DataFrame({'predictions':restored_output})
finall.to_csv('./robustness_cosmos.csv')