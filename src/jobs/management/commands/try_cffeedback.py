import collections

from django.core.management.base import BaseCommand
from pymining import seqmining, itemmining
from sklearn.externals import joblib

from src.core.core import get_encoded_logs
from src.explanation.lime_wrapper import lime_temporal_stability
from src.explanation.models import Explanation, ExplanationTypes
from src.explanation.temporal_stability import temporal_stability
from src.jobs.models import Job
from src.split.models import Split
from src.utils.prettyjson import prettyjson


class Command(BaseCommand):
    help = 'tries to deliver an explanation of a random prediction of the trained model'

    def handle(self, *args, **kwargs):
        TARGET_JOB = 439
        SPLITID = 155
        job_obj = Job.objects.filter(pk=TARGET_JOB)[0]
        split_obj = Split.objects.filter(pk=SPLITID)[0]

        training_df, test_df = get_encoded_logs(job_obj)

        test_df1 = test_df.copy()
        test_df2 = test_df.copy()
        test_df3 = test_df.copy()

        # todo: retrieve lime explanation

        # RETRIEVE&SAVE TS
        ts_exp_job, _ = Explanation.objects.get_or_create(type=ExplanationTypes.TEMPORAL_STABILITY.value, split=split_obj, predictive_model=job_obj.predictive_model, job=job_obj)
        ts = temporal_stability(ts_exp_job, training_df, test_df1, explanation_target=None)

        # RETRIEVE&SAVE LIMETS
        limets_exp_job, _ = Explanation.objects.get_or_create(type=ExplanationTypes.LIME.value, split=split_obj, predictive_model=job_obj.predictive_model, job=job_obj)
        lime_ts = lime_temporal_stability(limets_exp_job, training_df, test_df2, explanation_target=None)

        # SAVE GOLD
        gold = test_df3[['trace_id', 'label']]


        # todo: retrieve confusion matrix

        ts = {
            asdf: {
                uuu + '1' if uuu[-1:] == '_' else uuu: ts[asdf][uuu]
                for uuu in ts[asdf]
            }
            for asdf in ts
        }
        lime_ts = {
            asdf: {
                uuu + '1' if uuu[-1:] == '_' else uuu: lime_ts[asdf][uuu]
                for uuu in lime_ts[asdf]
            }
            for asdf in lime_ts
        }

        trace_ids = set(gold['trace_id'])
        confusion_matrix = {
            'tp': [
                str(tid)
                for tid in trace_ids
                if str(tid) in ts and
                   ts[str(tid)]['prefix_' + str(len(ts[str(tid)]))]['predicted'] == 'true' and
                   ts[str(tid)]['prefix_' + str(len(ts[str(tid)]))]['predicted'] == (
                       'true' if gold[gold['trace_id'] == tid]['label'].values[0] else 'false')
            ],
            'tn': [
                str(tid)
                for tid in trace_ids
                if str(tid) in ts and
                   ts[str(tid)]['prefix_' + str(len(ts[str(tid)]))]['predicted'] == 'false' and
                   ts[str(tid)]['prefix_' + str(len(ts[str(tid)]))]['predicted'] == (
                       'true' if gold[gold['trace_id'] == tid]['label'].values[0] else 'false')
            ],
            'fp': [
                str(tid)
                for tid in trace_ids
                if str(tid) in ts and
                   ts[str(tid)]['prefix_' + str(len(ts[str(tid)]))]['predicted'] == 'true' and
                   ts[str(tid)]['prefix_' + str(len(ts[str(tid)]))]['predicted'] != (
                       'true' if gold[gold['trace_id'] == tid]['label'].values[0] else 'false')
            ],
            'fn': [
                str(tid)
                for tid in trace_ids
                if str(tid) in ts and
                   ts[str(tid)]['prefix_' + str(len(ts[str(tid)]))]['predicted'] == 'false' and
                   ts[str(tid)]['prefix_' + str(len(ts[str(tid)]))]['predicted'] != (
                       'true' if gold[gold['trace_id'] == tid]['label'].values[0] else 'false')
            ]
        }

        limefeats = {
            k: {
                key: [
                    element
                    for element in sorted(
                        [(pref, lime_ts[key]['prefix_' + str(job_obj.encoding.prefix_length)][pref]['value'],
                          lime_ts[key]['prefix_' + str(job_obj.encoding.prefix_length)][pref]['importance']) for pref in lime_ts[key]['prefix_' + str(job_obj.encoding.prefix_length)]],
                        key=lambda x: (x[2], x[1]),
                        reverse=True if k in ['tp', 'fp'] else False
                        # reverse order of lime values if the prediction is negative
                    )
                ]
                for key in confusion_matrix[k]
                if 'prefix_' + str(job_obj.encoding.prefix_length) in lime_ts[key]
            }
            for k in confusion_matrix
        }

        freq_seqs = {
            'tp': {},
            'tn': {},
            'fp': {},
            'fn': {}
        }

        # todo: retrive patterns
        CONFUSION_MATRIX = ['tp', 'tn', 'fp', 'fn']

        LIMEFEATS = {
            'abs_lime': False,
            'tp': 0.2,
            'tn': 0.2,
            'fp': 0.2,
            'fn': 0.2,
            'top': 10,
            'outputfile': None
        }
        FREQ_SEQS = {
            'tp': 10,
            'tn': 10,
            'fp': 10,
            'fn': 10,
            'top': 15,
            'outputfile': None,
            'RECOMPUTEDoutputfile': None,
        }
        ABSENCE = {
            'tp': 0.1,
            'tn': 0.1,
            'fp': 0.1,
            'fn': 0.1,
            'ABSENCEoutputfile': None
        }

        MINING_METHOD = 'item_mining'

        print(
            'Initial CONFUSION MATRIX:\n',
            *['\tlimefeats ' + KEY + ':' + str(len(limefeats[KEY])) for KEY in CONFUSION_MATRIX],
            '\n',
            *['\tfreq_seqs ' + KEY + ':' + str(len(freq_seqs[KEY])) for KEY in CONFUSION_MATRIX]
        )

        available_values = {}
        for KEY in CONFUSION_MATRIX:
            available_values[KEY] = {}
            for tid in limefeats[KEY]:
                for event in limefeats[KEY][tid]:
                    if event[0].split('_')[0] not in available_values[KEY]:
                        available_values[KEY][event[0].split('_')[0]] = set()
                    available_values[KEY][event[0].split('_')[0]].add(event[1])

        filtered_limefeats = {
            KEY: {
                tid: [
                    event
                    for event in limefeats[KEY][tid]
                    if (
                           (not LIMEFEATS['abs_lime']) and
                           ((KEY in ['tp', 'fp'] and event[2] >= LIMEFEATS[KEY]) or
                            (KEY in ['tn', 'fn'] and event[2] <= -LIMEFEATS[KEY]))
                       ) or (
                           LIMEFEATS['abs_lime'] and abs(event[2]) >= LIMEFEATS[KEY]
                       )
                ]
                for tid in limefeats[KEY]
            }
            for KEY in CONFUSION_MATRIX
        }

        prefiltered_limefeats = {
            KEY: {
                tid: [
                    event
                    for event in limefeats[KEY][tid]
                    if (
                           (not LIMEFEATS['abs_lime']) and
                           ((KEY in ['tp', 'fp'] and event[2] >= LIMEFEATS[KEY]) or
                            (KEY in ['tn', 'fn'] and event[2] <= -LIMEFEATS[KEY]))
                       ) or (
                           LIMEFEATS['abs_lime'] and abs(event[2]) >= LIMEFEATS[KEY]
                       )
                ]
                for tid in limefeats[KEY]
            }
            for KEY in CONFUSION_MATRIX
        }

        filtered_limefeats_mine = {
            KEY: {
                tid:
                    prefiltered_limefeats[KEY][tid][0:LIMEFEATS['top']]
                for tid in prefiltered_limefeats[KEY]
            }
            for KEY in CONFUSION_MATRIX
        }

        for KEY in CONFUSION_MATRIX:
            for k in list(filtered_limefeats[KEY]):
                if len(filtered_limefeats[KEY][k]) == 0:
                    del filtered_limefeats[KEY][k]

        def tassellate_numbers(element):
            element = str(element)
            return str(element).split('.')[0][0] + '0' \
                if \
                '.' in str(element) \
                and \
                len(str(element)) <= 5 \
                else \
                str(element).split('.')[0][0:4] \
                    if \
                    '.' in str(element) \
                    and \
                    len(str(element)) >= 10 \
                    else \
                    element

        def retrieve_right_len(element, available_values):
            if '_' in element:
                return len(available_values[element.split('_')[0]])
            else:
                retval = []
                for attribute in available_values:
                    if any([str(element) == str(tassellate_numbers(value)) for value in available_values[attribute]]):
                        retval += [len(available_values[attribute])]
                return max(retval)

        def weight_freq_seqs(KEY, available_values, element, limefeats):
            print(element[0])
            print(
                'frequency:', element[1], ' * ',
                'len w/out absences: ', len([el for el in element[0] if 'absence' not in el]), ' * ',
                'sum of enumerator of possible values: ',
                sum([retrieve_right_len(el, available_values[KEY]) for el in element[0] if 'absence' not in el]), ' / ',
                'amount of examples in the field of confusion matrix: ', len(limefeats[KEY]), ' = ', (
                                                                                                         element[1] *
                                                                                                         len([el for el
                                                                                                              in
                                                                                                              element[0]
                                                                                                              if
                                                                                                              'absence' not in el]) *
                                                                                                         sum([
                                                                                                                 retrieve_right_len(
                                                                                                                     el,
                                                                                                                     available_values[
                                                                                                                         KEY])
                                                                                                                 for el
                                                                                                                 in
                                                                                                                 element[
                                                                                                                     0]
                                                                                                                 if
                                                                                                                 'absence' not in el])
                                                                                                     ) / len(
                    limefeats[KEY]))
            return (
                       element[1]  # *
                       # len([el for el in element[0] if 'absence' not in el]) *
                       # sum([retrieve_right_len(el, available_values[KEY]) for el in element[0] if 'absence' not in el])
                   ) / len(limefeats[KEY])

        filtered_freq_seqs_old = {
            KEY: sorted([
                element
                for element in freq_seqs[KEY]
                if weight_freq_seqs(KEY, available_values, element, limefeats) >= FREQ_SEQS[KEY]
            ], key=lambda x: x[1], reverse=True)
            for KEY in CONFUSION_MATRIX
        }

        prefiltered_freq_seqs = {
            KEY: sorted([
                element
                for element in freq_seqs[KEY]
                if weight_freq_seqs(KEY, available_values, element, limefeats) >= FREQ_SEQS[KEY]
            ], key=lambda x: x[1], reverse=True)
            for KEY in CONFUSION_MATRIX
        }

        #todo: is this the actual topK?
        filtered_freq_seqs = {
            KEY:
                prefiltered_freq_seqs[KEY][0:FREQ_SEQS['top']]
            for KEY in CONFUSION_MATRIX
        }

        print(
            'CONFUSION MATRIX after filtering:\n',
            *['\tlimefeats ' + KEY + ':' + str(len(filtered_limefeats[KEY])) for KEY in CONFUSION_MATRIX],
            '\n',
            *['\tfreq_seqs ' + KEY + ':' + str(len(filtered_freq_seqs[KEY])) for KEY in CONFUSION_MATRIX]
        )

        def printout_freq_seqs(output_obj, output_file, maxlinelength=5000):
            with open(output_file, 'w+') as f:
                f.write(prettyjson(output_obj, maxlinelength=maxlinelength))

        if (LIMEFEATS['outputfile'] is not None or FREQ_SEQS['outputfile'] is not None):
            print('Start saving results..')
            if (LIMEFEATS['outputfile'] is not None):
                printout_freq_seqs(filtered_limefeats, LIMEFEATS['outputfile'], maxlinelength=5000)
            if (FREQ_SEQS['outputfile'] is not None):
                printout_freq_seqs(filtered_freq_seqs, FREQ_SEQS['outputfile'], maxlinelength=200)
            print('Results saved.')
        else:
            print('FILTERED_LIMEFEATS:\n', filtered_limefeats)
            print('FILTERED_FREQ_SEQS:\n', filtered_freq_seqs)

        print('Computing absence...')

        attributes = {}
        for KEY in CONFUSION_MATRIX:
            for tid in limefeats[KEY]:
                for event in limefeats[KEY][tid]:
                    attribute_name = event[0]
                    if attribute_name not in attributes:
                        attributes[attribute_name] = set()
                    attributes[attribute_name].add(event[1])

        attributes_occurrences = {
            'tp': collections.Counter(),
            'fp': collections.Counter(),
            'tn': collections.Counter(),
            'fn': collections.Counter()
        }

        for KEY in CONFUSION_MATRIX:
            found_stuff = []
            for tid in limefeats[KEY]:
                for event in limefeats[KEY][tid]:
                    found_stuff += [tassellate_numbers(event[1])]

            attributes_occurrences[KEY].update(found_stuff)

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
        print('Absence computed.')
        print(
            'The absence AFTER filtering is:\n',
            prettyjson(characterised_attributes_occurrences)
        )

        print('RE-computing the sequence pattern result after applying the thresholds...')

        static_attr = [
            #    'Age',
            #    'ClaimValue',
            #    'CType',
            #    'ClType',
            #    'PClaims',
        ]
        limefeats_static_dinamic = {}
        for KEY in CONFUSION_MATRIX:
            limefeats_static_dinamic[KEY] = {}
            for tid in filtered_limefeats[KEY]:
                limefeats_static_dinamic[KEY][tid] = {
                    'static': [],
                    'dynamic': [
                        att
                        for att in filtered_limefeats[KEY][tid]
                        if not any([att[0].startswith(static_att) for static_att in static_attr])
                    ]
                }
                current_static_attributes = [
                    att
                    for att in filtered_limefeats[KEY][tid]
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
                            limefeats_static_dinamic[KEY][tid]['static'] += [max(curr_attributes, key=lambda x: x[2])]
                        elif KEY in ['tn', 'fn']:
                            limefeats_static_dinamic[KEY][tid]['static'] += [max(curr_attributes, key=lambda x: x[2])]
                        else:
                            print('Something bad happened')

        dynamic_data = {
            KEY: {
                tid: [
                    # (element[0].split('_')[0] + '_' +  element[1])
                    (element[0] + '_' + element[1])
                    for element in sorted(
                        [k for k in limefeats_static_dinamic[KEY][tid]['dynamic']],
                        # key=lambda x: (x[0].split('_')[1], x[0].split('_')[0])
                        key=lambda x: x[0]
                    )
                ]
                for tid in limefeats_static_dinamic[KEY]
                if len(limefeats_static_dinamic[KEY][tid]['dynamic']) > 0
            }
            for KEY in CONFUSION_MATRIX
        }

        static_data = {
            KEY: {
                tid: [
                    (element[0].split('_')[0] + '_' + tassellate_numbers(element[1]))
                    # (element[0] + '_' + tassellate_numbers(element[1]))
                    for element in sorted(
                        [k for k in limefeats_static_dinamic[KEY][tid]['static']],
                        key=lambda x: (x[0].split('_')[1], x[0].split('_')[0])
                    )
                ]
                for tid in limefeats_static_dinamic[KEY]
                if len(limefeats_static_dinamic[KEY][tid]['static']) > 0
            }
            for KEY in CONFUSION_MATRIX
        }

        data = {}
        for KEY in CONFUSION_MATRIX:
            data[KEY] = {}
            for tid in limefeats[KEY]:
                if tid in static_data[KEY] and tid in dynamic_data[KEY]:
                    data[KEY][tid] = static_data[KEY][tid] + dynamic_data[KEY][tid]
                elif tid in static_data[KEY]:
                    data[KEY][tid] = static_data[KEY][tid]
                elif tid in dynamic_data[KEY]:
                    data[KEY][tid] = dynamic_data[KEY][tid]

        if (MINING_METHOD == 'seq_mining'):
            freq_seqs_after_filter = {
                'tp': sorted(seqmining.freq_seq_enum([data['tp'][tid] for tid in data['tp']], 2)),
                'tn': sorted(seqmining.freq_seq_enum([data['tn'][tid] for tid in data['tn']], 2)),
                'fp': sorted(seqmining.freq_seq_enum([data['fp'][tid] for tid in data['fp']], 2)),
                'fn': sorted(seqmining.freq_seq_enum([data['fn'][tid] for tid in data['fn']], 2)),
            }
        if (MINING_METHOD == 'item_mining'):
            freq_seqs_after_filter = {
                'tp': itemmining.relim(itemmining.get_relim_input([data['tp'][tid] for tid in data['tp']]), min_support=2),
                'tn': itemmining.relim(itemmining.get_relim_input([data['tn'][tid] for tid in data['tn']]), min_support=2),
                'fp': itemmining.relim(itemmining.get_relim_input([data['fp'][tid] for tid in data['fp']]), min_support=2),
                'fn': itemmining.relim(itemmining.get_relim_input([data['fn'][tid] for tid in data['fn']]), min_support=2),
            }

            freq_seqs_after_filter = {
                KEY: [
                    (tuple(element), freq_seqs_after_filter[KEY][element])
                    for element in freq_seqs_after_filter[KEY]]
                for KEY in CONFUSION_MATRIX
            }

        filtered_freq_seqs_after_filter_old = {
            KEY: sorted([
                [element[0], weight_freq_seqs(KEY, available_values, element, limefeats)]
                for element in freq_seqs_after_filter[KEY]
                if weight_freq_seqs(KEY, available_values, element, limefeats) >= FREQ_SEQS[KEY]
            ], key=lambda x: x[1], reverse=True)
            for KEY in CONFUSION_MATRIX
        }

        # todo: filter topK
        filtered_freq_seqs_after_filter = {
            KEY:
                filtered_freq_seqs_after_filter_old[KEY][0:FREQ_SEQS['top']]
            for KEY in CONFUSION_MATRIX
        }

        print('Sequence pattern recomputed successfully.')

        if (FREQ_SEQS['outputfile'] is not None):
            print('Start saving results..')
            printout_freq_seqs(filtered_freq_seqs_after_filter, FREQ_SEQS['RECOMPUTEDoutputfile'], maxlinelength=200)
            print('Results saved.')
        else:
            print('RECOMPUTED_FREQ_SEQS:\n', prettyjson(filtered_freq_seqs_after_filter, maxlinelength=200))

        print('Done, cheers!')
        # return confusion_matrix, data, freq_seqs_after_filter, filtered_freq_seqs_after_filter
