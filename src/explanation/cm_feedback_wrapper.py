import collections

from pymining import seqmining, itemmining

from src.encoding.common import retrieve_proper_encoder
from src.encoding.encoder import PREFIX_
from src.explanation.lime_wrapper import lime_temporal_stability
from src.explanation.models import Explanation, ExplanationTypes
from src.explanation.temporal_stability import temporal_stability

from ansible.module_utils.parsing.convert_bool import boolean


def retrieve_temporal_stability(training_df, test_df, job_obj, split_obj):
    ts_exp_job, _ = Explanation.objects.get_or_create(
        type=ExplanationTypes.TEMPORAL_STABILITY.value,
        split=split_obj,
        predictive_model=job_obj.predictive_model,
        job=job_obj
    )
    ts = temporal_stability(ts_exp_job, training_df, test_df, explanation_target=None)
    ts = {
        trace_id: {
            element + '1' if element[-1:] == '_' else element: ts[trace_id][element]
            for element in ts[trace_id]
        }
        for trace_id in ts
    }
    return ts


def retrieve_lime_ts(training_df, test_df, job_obj, split_obj):
    limets_exp_job, _ = Explanation.objects.get_or_create(
        type=ExplanationTypes.LIME.value,
        split=split_obj,
        predictive_model=job_obj.predictive_model,
        job=job_obj
    )
    lime_ts = lime_temporal_stability(limets_exp_job, training_df, test_df, explanation_target=None)
    lime_ts = {
        trace_id: {
            element + '1' if element[-1:] == '_' else element: lime_ts[trace_id][element]
            for element in lime_ts[trace_id]
        }
        for trace_id in lime_ts
    }
    return lime_ts


def compute_confusion_matrix(ts, gold, job_obj):
    encoder = retrieve_proper_encoder(job_obj)
    encoder.decode(df=gold, encoding=job_obj.encoding)
    trace_ids = set(gold['trace_id'])
    confusion_matrix = {
        'tp': [
            str(trace_id)
            for trace_id in trace_ids
            if (str(trace_id) in ts) and
               (ts[str(trace_id)][PREFIX_ + str(len(ts[str(trace_id)]))]['predicted'] == 'true') and
               (ts[str(trace_id)][PREFIX_ + str(len(ts[str(trace_id)]))]['predicted'] ==
                ('true' if boolean(gold[gold['trace_id'] == trace_id]['label'].values[0]) else 'false'))
        ],
        'tn': [
            str(trace_id)
            for trace_id in trace_ids
            if (str(trace_id) in ts) and
               (ts[str(trace_id)][PREFIX_ + str(len(ts[str(trace_id)]))]['predicted'] == 'false') and
               (ts[str(trace_id)][PREFIX_ + str(len(ts[str(trace_id)]))]['predicted'] ==
                ('true' if boolean(gold[gold['trace_id'] == trace_id]['label'].values[0]) else 'false'))
        ],
        'fp': [
            str(trace_id)
            for trace_id in trace_ids
            if (str(trace_id) in ts) and
               (ts[str(trace_id)][PREFIX_ + str(len(ts[str(trace_id)]))]['predicted'] == 'true') and
               (ts[str(trace_id)][PREFIX_ + str(len(ts[str(trace_id)]))]['predicted'] !=
                ('true' if boolean(gold[gold['trace_id'] == trace_id]['label'].values[0]) else 'false'))
        ],
        'fn': [
            str(trace_id)
            for trace_id in trace_ids
            if (str(trace_id) in ts) and
               (ts[str(trace_id)][PREFIX_ + str(len(ts[str(trace_id)]))]['predicted'] == 'false') and
               (ts[str(trace_id)][PREFIX_ + str(len(ts[str(trace_id)]))]['predicted'] !=
                ('true' if boolean(gold[gold['trace_id'] == trace_id]['label'].values[0]) else 'false'))
        ]
    }
    return confusion_matrix


def tassellate_numbers(element):  # todo: not futureproof
    element = str(element)
    return element.split('.')[0][0] + '0' if '.' in element and len(element) <= 5 \
        else \
        element.split('.')[0][0:4] if '.' in element and len(element) >= 10 \
            else \
            element


# def retrieve_right_len(element, available_values):
#     if '_' in element:
#         return len(available_values[element.split('_')[0]])
#     else:
#         retval = []
#         for attribute in available_values:
#             if any([str(element) == str(tassellate_numbers(value)) for value in available_values[attribute]]):
#                 retval += [len(available_values[attribute])]
#         return max(retval)


def weight_freq_seqs(KEY, element, limefeats, available_values):
    return (
               element[1]  # *
               # len([el for el in element[0] if 'absence' not in el]) *
               # sum([retrieve_right_len(el, available_values[KEY]) for el in element[0] if 'absence' not in el])
    ) / len(limefeats[KEY])


def mine_patterns(data, MINING_METHOD, CONFUSION_MATRIX):
    if (MINING_METHOD == 'seq_mining'):
        mined_patterns = {
            KEY: sorted(seqmining.freq_seq_enum([data[KEY][trace_id] for trace_id in data[KEY]], min_support=2))
            for KEY in CONFUSION_MATRIX
        }
    if (MINING_METHOD == 'item_mining'):
        mined_patterns_to_be_preprocessed = {
            KEY: itemmining.relim(itemmining.get_relim_input([data[KEY][trace_id] for trace_id in data[KEY]]), min_support=2)
            for KEY in CONFUSION_MATRIX
        }

        mined_patterns = {
            KEY: [
                (tuple(element), mined_patterns_to_be_preprocessed[KEY][element])
                for element in mined_patterns_to_be_preprocessed[KEY]]
            for KEY in CONFUSION_MATRIX
        }
    return mined_patterns


def filter_lime_features(limefeats, LIMEFEATS, CONFUSION_MATRIX):
    filtered_limefeats = {
        KEY: {
            trace_id: [
                event
                for event in limefeats[KEY][trace_id]
                if (
                       (not LIMEFEATS['abs_lime']) and
                       ((KEY in ['tp', 'fp'] and event[2] >= LIMEFEATS[KEY]) or
                        (KEY in ['tn', 'fn'] and event[2] <= -LIMEFEATS[KEY]))
                   ) or (
                       LIMEFEATS['abs_lime'] and abs(event[2]) >= LIMEFEATS[KEY]
                   )
            ]
            for trace_id in limefeats[KEY]
        }
        for KEY in CONFUSION_MATRIX
    }

    for KEY in CONFUSION_MATRIX:
        for trace_id in list(filtered_limefeats[KEY]):
            if len(filtered_limefeats[KEY][trace_id]) == 0:
                del filtered_limefeats[KEY][trace_id]

    return filtered_limefeats


def compute_attributes(CONFUSION_MATRIX, limefeats):

    attributes_occurrences = {
        KEY: collections.Counter([
            tassellate_numbers(event[1])
            for trace_id in limefeats[KEY]
            for event in limefeats[KEY][trace_id]
        ])
        for KEY in CONFUSION_MATRIX
    }

    attributes = {}
    for KEY in CONFUSION_MATRIX:
        for trace_id in limefeats[KEY]:
            for event in limefeats[KEY][trace_id]:
                attribute_name = event[0]
                if attribute_name not in attributes:
                    attributes[attribute_name] = set()
                attributes[attribute_name].add(event[1])

    characterised_attributes_occurrences = {}
    for KEY in CONFUSION_MATRIX:
        characterised_attributes_occurrences[KEY] = {}
        for attribute in attributes:
            if attribute not in characterised_attributes_occurrences[KEY]:
                characterised_attributes_occurrences[KEY][attribute] = dict()
            for attr in attributes[attribute]:
                characterised_attributes_occurrences[KEY][attribute][tassellate_numbers(attr)] = 0
    for KEY in CONFUSION_MATRIX:
        for occ in attributes_occurrences[KEY]:
            for attr in characterised_attributes_occurrences[KEY]:
                if occ in characterised_attributes_occurrences[KEY][attr]:
                    characterised_attributes_occurrences[KEY][attr][occ] = attributes_occurrences[KEY][occ]
        for attr in characterised_attributes_occurrences[KEY]:
            characterised_attributes_occurrences[KEY][attr]['Total'] = sum([
                characterised_attributes_occurrences[KEY][attr][element]
                for element in characterised_attributes_occurrences[KEY][attr]
            ])
    return attributes, attributes_occurrences, characterised_attributes_occurrences


def compute_data(CONFUSION_MATRIX, limefeats, filtered_limefeats):
    static_attr = [  # todo: find a way to compute auto-magically
        #    'Age',
        #    'ClaimValue',
        #    'CType',
        #    'ClType',
        #    'PClaims',
    ]
    limefeats_static_dinamic = {}
    for KEY in CONFUSION_MATRIX:
        limefeats_static_dinamic[KEY] = {}
        for trace_id in filtered_limefeats[KEY]:
            limefeats_static_dinamic[KEY][trace_id] = {
                'static': [],
                'dynamic': [
                    att
                    for att in filtered_limefeats[KEY][trace_id]
                    if not any([att[0].startswith(static_att) for static_att in static_attr])
                ]
            }
            current_static_attributes = [
                att
                for att in filtered_limefeats[KEY][trace_id]
                if any([att[0].startswith(static_att) for static_att in static_attr])
            ]
            for s_attr in static_attr:
                curr_attributes = [
                    att
                    for att in current_static_attributes
                    if att[0].startswith(s_attr)
                ]
                if len(curr_attributes) > 0:
                    if KEY in ['tp', 'fp']:
                        limefeats_static_dinamic[KEY][trace_id]['static'] += [max(curr_attributes, key=lambda x: x[2])]
                    elif KEY in ['tn', 'fn']:
                        limefeats_static_dinamic[KEY][trace_id]['static'] += [max(curr_attributes, key=lambda x: x[2])]
                    else:
                        print('Something bad happened')

    dynamic_data = {
        KEY: {
            trace_id: [
                # (element[0].split('_')[0] + '_' +  element[1])
                (element[0] + '_' + element[1])
                for element in sorted(
                    [k for k in limefeats_static_dinamic[KEY][trace_id]['dynamic']],
                    # key=lambda x: (x[0].split('_')[1], x[0].split('_')[0])
                    key=lambda x: x[0]
                )
            ]
            for trace_id in limefeats_static_dinamic[KEY]
            if len(limefeats_static_dinamic[KEY][trace_id]['dynamic']) > 0
        }
        for KEY in CONFUSION_MATRIX
    }

    static_data = {
        KEY: {
            trace_id: [
                (element[0].split('_')[0] + '_' + tassellate_numbers(element[1]))
                # (element[0] + '_' + tassellate_numbers(element[1]))
                for element in sorted(
                    [k for k in limefeats_static_dinamic[KEY][trace_id]['static']],
                    key=lambda x: (x[0].split('_')[1], x[0].split('_')[0])
                )
            ]
            for trace_id in limefeats_static_dinamic[KEY]
            if len(limefeats_static_dinamic[KEY][trace_id]['static']) > 0
        }
        for KEY in CONFUSION_MATRIX
    }

    return {
        KEY: {
            trace_id: static_data[KEY].get(trace_id, []) + dynamic_data[KEY].get(trace_id, [])
            for trace_id in limefeats[KEY]
        }
        for KEY in CONFUSION_MATRIX
    }


def process_lime_features(lime_ts, confusion_matrix, CONFUSION_MATRIX, prefix_length):
    return {
        KEY: {
            trace_id: [
                element
                for element in sorted([
                        (
                            pref,
                            lime_ts[trace_id][PREFIX_ + str(prefix_length)][pref]['value'],
                            lime_ts[trace_id][PREFIX_ + str(prefix_length)][pref]['importance']
                        )
                        for pref in lime_ts[trace_id][PREFIX_ + str(prefix_length)]
                    ],
                    key=lambda x: (x[2], x[1]),
                    reverse=True if KEY in ['tp', 'fp'] else False
                    # reverse order of lime values if the prediction is negative
                )
            ]
            for trace_id in confusion_matrix[KEY]
            if PREFIX_ + str(prefix_length) in lime_ts[trace_id]
        }
        for KEY in CONFUSION_MATRIX
    }


def explain(cffeedback_exp: Explanation, training_df, test_df, top_k, prefix_target):

    LIMEFEATS = {
        'abs_lime': False,
        'tp': 0.2,
        'tn': 0.2,
        'fp': 0.2,
        'fn': 0.2
    }
    FREQ_SEQS = {
        'tp': 0.1,
        'tn': 0.1,
        'fp': 0.1,
        'fn': 0.1
    }

    MINING_METHOD = 'item_mining' # 'seq_mining'

    CONFUSION_MATRIX = ['tp', 'tn', 'fp', 'fn']

    job_obj = cffeedback_exp.job
    split_obj = job_obj.split

    # todo: retrieve confusion matrix
    ts = retrieve_temporal_stability(training_df, test_df.copy(), job_obj, split_obj)
    confusion_matrix = compute_confusion_matrix(ts, gold=test_df[['trace_id', 'label']], job_obj=job_obj)

    lime_ts = retrieve_lime_ts(training_df, test_df.copy(), job_obj, split_obj)
    limefeats = process_lime_features(lime_ts, confusion_matrix, CONFUSION_MATRIX, job_obj.encoding.prefix_length)
    filtered_limefeats = filter_lime_features(limefeats, LIMEFEATS, CONFUSION_MATRIX)

    #todo MAYBE I SHOULD RETURN ALSO THIS? (it is pretty fitted on the bpi and drift data)
    # attributes, attributes_occurrences, characterised_attributes_occurrences = compute_attributes(CONFUSION_MATRIX, limefeats)

    data = compute_data(CONFUSION_MATRIX, limefeats, filtered_limefeats)
    frequent_patterns = mine_patterns(data, MINING_METHOD, CONFUSION_MATRIX)

    available_values = {}
    for KEY in CONFUSION_MATRIX:
        available_values[KEY] = {}
        for trace_id in limefeats[KEY]:
            for event in limefeats[KEY][trace_id]:
                if event[0].split('_')[0] not in available_values[KEY]:
                    available_values[KEY][event[0].split('_')[0]] = set()
                available_values[KEY][event[0].split('_')[0]].add(event[1])

    frequent_patterns_ordered = {
        KEY: sorted([
            [element[0], weight_freq_seqs(KEY, element, limefeats, available_values)]
            for element in frequent_patterns[KEY]
            if weight_freq_seqs(KEY, element, limefeats, available_values) >= FREQ_SEQS[KEY]
        ], key=lambda x: x[1], reverse=True)
        for KEY in CONFUSION_MATRIX
    }

    topK_frequent_patterns = {
        KEY: frequent_patterns_ordered[KEY][0:top_k]
        for KEY in CONFUSION_MATRIX
    }

    return {"confusion_matrix": confusion_matrix, "data": data,
            "freq_seqs_after_filter": frequent_patterns, "filtered_freq_seqs_after_filter": topK_frequent_patterns}
