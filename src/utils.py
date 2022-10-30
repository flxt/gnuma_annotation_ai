import argparse
import json
import os
import random
import uuid

from src import logwrapper, dispatcher


# Get arguments for training
def get_train_args(project_id: str, model_exists, data_path: str):
    arg_parser = argparse.ArgumentParser()

    #data pathes
    arg_parser.add_argument('--types_path', type=str, default=f'train/{data_path}/types.json')
    arg_parser.add_argument('--train_path', type=str, default=f'train/{data_path}/train.json')
    arg_parser.add_argument('--valid_path', type=str, default=f'train/{data_path}/train.json')

    # Preprocessing
    arg_parser.add_argument('--tokenizer_path', type=str, default='bert-base-cased')
    arg_parser.add_argument('--max_span_size', type=int, default=10)
    arg_parser.add_argument('--lowercase', action='store_true', default=False)
    arg_parser.add_argument('--sampling_processes', type=int, default=4)

    # Model / Training / Evaluation
    if not model_exists:
        arg_parser.add_argument('--model_path', type=str, default='bert-base-cased')
    else:
        arg_parser.add_argument('--model_path', type=str, default=f'models/{project_id}')

    arg_parser.add_argument('--model_type', type=str, default='spert')
    arg_parser.add_argument('--cpu', action='store_true', default=False)
    arg_parser.add_argument('--eval_batch_size', type=int, default=1)
    arg_parser.add_argument('--max_pairs', type=int, default=1000)
    arg_parser.add_argument('--rel_filter_threshold', type=float, default=0.4)
    arg_parser.add_argument('--size_embedding', type=int, default=25)
    arg_parser.add_argument('--prop_drop', type=float, default=0.1)
    arg_parser.add_argument('--freeze_transformer', action='store_true', default=False)
    arg_parser.add_argument('--no_overlapping', action='store_true', default=True)

    # Misc
    arg_parser.add_argument('--seed', type=int, default=None)
    arg_parser.add_argument('--cache_path', type=str, default=None)
    arg_parser.add_argument('--debug', action='store_true', default=False)

    #Logging
    arg_parser.add_argument('--label', type=str, default=project_id)
    arg_parser.add_argument('--log_path', type=str, default='./logs')
    arg_parser.add_argument('--store_predictions', action='store_true', default=False)
    arg_parser.add_argument('--store_examples', action='store_true', default=False)
    arg_parser.add_argument('--example_count', type=int, default=None)

    # Logging
    arg_parser.add_argument('--save_path', type=str, default=f'models/')
    arg_parser.add_argument('--init_eval', action='store_true', default=False)
    arg_parser.add_argument('--save_optimizer', action='store_true', default=True)
    arg_parser.add_argument('--train_log_iter', type=int, default=100)
    arg_parser.add_argument('--final_eval', action='store_true', default=False)
    arg_parser.add_argument('--do_eval', action='store_true', default=False)

    # Model / Training
    arg_parser.add_argument('--train_batch_size', type=int, default=1)
    arg_parser.add_argument('--epochs', type=int, default=1)
    arg_parser.add_argument('--neg_entity_count', type=int, default=100)
    arg_parser.add_argument('--neg_relation_count', type=int, default=100)
    arg_parser.add_argument('--lr', type=float, default=5e-5)
    arg_parser.add_argument('--lr_warmup', type=float, default=0.0)
    arg_parser.add_argument('--weight_decay', type=float, default=0.01)
    arg_parser.add_argument('--max_grad_norm', type=float, default=1.0)

    return arg_parser


# Get arguments for prediction
def get_pred_args(project_id: str, pred_id: str, doc_id: str):
    arg_parser = argparse.ArgumentParser()

    #data pathes
    arg_parser.add_argument('--types_path', type=str, default=f'pred/{pred_id}/types.json')

    # Preprocessing
    arg_parser.add_argument('--tokenizer_path', type=str, default='bert-base-cased')
    arg_parser.add_argument('--max_span_size', type=int, default=10)
    arg_parser.add_argument('--lowercase', action='store_true', default=False)
    arg_parser.add_argument('--sampling_processes', type=int, default=4)

    # Model / Training / Evaluation
    arg_parser.add_argument('--model_path', type=str, default=f'models/{project_id}')

    arg_parser.add_argument('--model_type', type=str, default='spert')
    arg_parser.add_argument('--cpu', action='store_true', default=False)
    arg_parser.add_argument('--eval_batch_size', type=int, default=1)
    arg_parser.add_argument('--max_pairs', type=int, default=1000)
    arg_parser.add_argument('--rel_filter_threshold', type=float, default=0.4)
    arg_parser.add_argument('--size_embedding', type=int, default=25)
    arg_parser.add_argument('--prop_drop', type=float, default=0.1)
    arg_parser.add_argument('--freeze_transformer', action='store_true', default=False)
    arg_parser.add_argument('--no_overlapping', action='store_true', default=True)

    # Misc
    arg_parser.add_argument('--seed', type=int, default=None)
    arg_parser.add_argument('--cache_path', type=str, default=None)
    arg_parser.add_argument('--debug', action='store_true', default=False)

    #Logging
    arg_parser.add_argument('--label', type=str, default=project_id)
    arg_parser.add_argument('--log_path', type=str, default='./logs')
    arg_parser.add_argument('--store_predictions', action='store_true', default=False)
    arg_parser.add_argument('--store_examples', action='store_true', default=False)
    arg_parser.add_argument('--example_count', type=int, default=None)

    # Logging
    arg_parser.add_argument('--save_path', type=str, default=f'models')
    arg_parser.add_argument('--init_eval', action='store_true', default=False)
    arg_parser.add_argument('--save_optimizer', action='store_true', default=True)
    arg_parser.add_argument('--train_log_iter', type=int, default=100)
    arg_parser.add_argument('--final_eval', action='store_true', default=False)
    arg_parser.add_argument('--do_eval', action='store_true', default=False)

    # Model / Training
    arg_parser.add_argument('--train_batch_size', type=int, default=1)
    arg_parser.add_argument('--epochs', type=int, default=1)
    arg_parser.add_argument('--neg_entity_count', type=int, default=100)
    arg_parser.add_argument('--neg_relation_count', type=int, default=100)
    arg_parser.add_argument('--lr', type=float, default=5e-5)
    arg_parser.add_argument('--lr_warmup', type=float, default=0.0)
    arg_parser.add_argument('--weight_decay', type=float, default=0.01)
    arg_parser.add_argument('--max_grad_norm', type=float, default=1.0)

    #pred
    # Input
    arg_parser.add_argument('--dataset_path', type=str, default=f'pred/{pred_id}/pred.json')
    arg_parser.add_argument('--predictions_path', type=str, default=f'pred/{pred_id}/results.json')
    arg_parser.add_argument('--spacy_model', type=str)

    return arg_parser


#get text of docuemnts from doc service
def get_doc(doc_id):
    doc = dispatcher.get_document(doc_id)

    sents = []
    for ele in doc['sentences']:
        tokens = []

        for tok in ele['tokens']:
            tokens.append(tok['token'])

        sents.append({
            'tokens': tokens,
            'entities': [],
            'relations': []
        })

    return sents

#make data ready for spert
def preprocess_train_data(data_id):
    with open(f'train/{data_id}/config.json') as f:
        config = json.load(f)

    # save training dat for rt_8s
    sample_size = 8
    sample_size = min(sample_size, len(config['train_data']) - 1) # true sample size

    # generate sample
    sample = random.sample(config['train_data'][0: -1], sample_size)

    # add newest train doc
    sample.append(config['train_data'][-1])

    logwrapper.debug(f'Tain Sample: {sample}')

    train_data = []

    # add all sampled training documents
    for sam in sample:
        doc = get_doc(sam['doc_id'])
        for index, sent in enumerate(doc):
            ents = []
            rels = []

            candidate_rel_ids = []
            for ele in sam['sentence_entities'][index]:
                dat = sam['entities'][ele]

                # entities for sentence
                ents.append({
                    'type': dat['type'],
                    'start': dat['start'],
                    'end': dat['end'],
                })

                # add to candidate relations
                candidate_rel_ids.extend(dat['relations'])

            # remove duplicates from candidate list
            list(dict.fromkeys(candidate_rel_ids))

            # add realtion but only if booth ents in this sentence
            for rel_id in candidate_rel_ids:
                rel = sam['relations'][rel_id]
                head = sam['entities'][rel['head']]
                tail = sam['entities'][rel['tail']]

                # add if same sentence id
                if head['sentenceIndex'] == tail['sentenceIndex']:
                    head_index = -1
                    tail_index = -1

                    # find head and tail index
                    for i, ent in enumerate(ents):
                        if head['start'] == ent['start'] and head['end'] == ent['end'] and head['type'] == ent['type']:
                            head_index = i
                        if tail['start'] == ent['start'] and tail['end'] == ent['end'] and tail['type'] == ent['type']:
                            tail_index = i

                    rels.append({
                        'type': rel['type'],
                        'head': head_index,
                        'tail': tail_index
                    })

            train_data.append({
                'tokens': sent['tokens'],
                'entities': ents,
                'relations': rels
            })

    with open(f'train/{data_id}/train.json', 'w') as f:
        json.dump(train_data, f)


    # types data
    ents = {}
    for ele in config['entity_types']:
        ents[ele] = {
            'short': ele,
            'verbose': ele
        }

    rels = {}
    for ele in config['relation_types']:
        rels[ele] = {
            'short': ele,
            'verbose': ele,
            'symmetric': False
        }

    types = {
        'entities': ents,
        'relations': rels
    }

    with open(f'train/{data_id}/types.json', 'w') as f:
        json.dump(types, f)

#make data ready for spert
def preprocess_pred_data(data_id):
    with open(f'pred/{data_id}/config.json') as f:
        config = json.load(f)

    # pred data
    doc = get_doc(config['document_id'])
    with open(f'pred/{data_id}/pred.json', 'w') as f:
        json.dump(doc, f)

    # types data
    ents = {}
    for ele in config['entity_types']:
        ents[ele] = {
            'short': ele,
            'verbose': ele
        }

    rels = {}
    for ele in config['relation_types']:
        rels[ele] = {
            'short': ele,
            'verbose': ele,
            'symmetric': False
        }

    types = {
        'entities': ents,
        'relations': rels
    }

    with open(f'pred/{data_id}/types.json', 'w') as f:
        json.dump(types, f)

    #return project id ansd doc id
    return config['project_id'], config['document_id']


# remove overlapping entities and relations
def remove_overlapping(ents, rels):
    new_ents = []
    new_rels = []

    cut = []
    #remove overlapping entities
    for index, ent in enumerate(ents):
        add = True
        for comp_ent in new_ents:
            if ((comp_ent['start'] <= ent['start'] < comp_ent['end']) or
                    (comp_ent['start'] < ent['end'] <= comp_ent['end'])):
                add = False

        if add:
            new_ents.append(ent)
        else:
            cut.append(index)

    #remove relations with overlapping entities
    for rel in rels:
        if rel['head'] not in cut and rel['tail'] not in cut:
            rel['head'] -= sum(x < rel['head'] for x in cut)
            rel['tail'] -= sum(x < rel['tail'] for x in cut)
            new_rels.append(rel)

    return new_ents, new_rels


# serialize and convert prediction to backend format
def serialize_prediction(data):
    entities = {}
    sentence_entities = []
    relations = {}

    for index, ele in enumerate(data):
        ents = []

        nov_ents, nov_rels = remove_overlapping(ele['entities'], ele['relations'])

        for ent in nov_ents:
            new_id = str(uuid.uuid4())

            #add to sentence entities
            ents.append(new_id)

            #creat entity
            entities[new_id] = {
                'id': new_id,
                'sentenceIndex': index,
                'start': ent['start'],
                'end': ent['end'],
                'type': ent['type'],
                'relations': []
            }

        sentence_entities.append(ents)

        for rel in nov_rels:
            new_id = str(uuid.uuid4())

            #add to relations
            entities[ents[rel['head']]]['relations'].append(new_id)
            entities[ents[rel['tail']]]['relations'].append(new_id)

            # create relation
            relations[new_id] = {
                'id': new_id,
                'head': ents[rel['head']],
                'tail': ents[rel['tail']],
                'type': rel['type']
            }

    out = {
        'recEntities': entities,
        'recSentenceEntities': sentence_entities,
        'recRelations': relations
    }

    return out
