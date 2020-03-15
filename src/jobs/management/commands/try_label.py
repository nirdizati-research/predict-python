import os
from django.core.management.base import BaseCommand

from pm4py.objects.log.importer.xes import factory as xes_import_factory
from pm4py.objects.log.exporter.xes import factory as xes_exporter

from pm4py.objects.log import log as lg


def contains_event(case: lg.Trace, event_name):
    found = False
    for event_index, event in enumerate(case):
        if event["concept:name"] == event_name:
            found = True
            break
    return found


class Command(BaseCommand):
    help = 'tries to deliver an explanation of a random prediction of the trained model'

    def handle(self, *args, **kwargs):
        log_path = os.path.join("", "",
                                "/mnt/c/Users/MusabirMusabayli/Desktop/Projects/DATASETS/train_explainability_1.xes")
        log_path
        log = xes_import_factory.apply(log_path)
        labelled_log = self.label_log(log)
        labelled_log_path = os.path.join("", "", "/mnt/c/Users/MusabirMusabayli/Desktop/Projects/DATASETS/train_explainability_1_filter_2_result.xes")

        xes_exporter.export_log(labelled_log, labelled_log_path)

    def label_log(self, log):
        labelled_log = lg.EventLog()
        self.attributes = log._attributes
        positive = 0
        negative = 0
        for case_index, case in enumerate(log):
            labelled_case = case
            first_event = case[0]
            attributes = labelled_case._get_attributes()
            if int(first_event["Age"]) < 60:
                if first_event["ClType"] == 'H12' or first_event["ClType"] == 'H14' or first_event["ClType"] == 'H15':
                    attributes.update(label="true")
                    positive += 1
                else:
                    attributes.update(label="false")
                    negative += 1
            else:
                attributes.update(label="false")
                negative += 1

            labelled_case._set_attributes(attributes)
            labelled_log.append(labelled_case)
        print('positive: ', str(positive), '; negative: ', str(negative))

        return labelled_log


# train 1 3rd level
#     def label_log(self, log):
#         labelled_log = lg.EventLog()
#         self.attributes = log._attributes
#         positive = 0
#         negative = 0
#         for case_index, case in enumerate(log):
#             labelled_case = case
#             first_event = case[0]
#             attributes = labelled_case._get_attributes()
#             if first_event['PClaims'] == 'No':
#                 if int(first_event["Age"]) <= 56:
#                     if first_event["CType"] == 'Regular' or first_event["CType"] == 'Silver' or first_event["CType"] == 'Gold':
#                         attributes.update(label="false")
#                         negative += 1
#                     else:
#                         attributes.update(label="true")
#                         positive += 1
#                 else:
#                     attributes.update(label="true")
#                     positive += 1
#             else:
#                 attributes.update(label="true")
#                 positive += 1
#             labelled_case._set_attributes(attributes)
#             labelled_log.append(labelled_case)
#         print('positive: ', str(positive), '; negative: ', str(negative))
#
#         return labelled_log


# train 2 2nd level
#
#
# def label_log(self, log):
#     labelled_log = lg.EventLog()
#     self.attributes = log._attributes
#     positive = 0
#     negative = 0
#     for case_index, case in enumerate(log):
#         labelled_case = case
#         first_event = case[0]
#         attributes = labelled_case._get_attributes()
#         if first_event["CType"] == 'Regular' or first_event["CType"] == 'VIP':
#             if first_event["ClType"] == 'H11' or first_event["ClType"] == 'H12' or first_event["ClType"] == 'H13':
#                 attributes.update(label="false")
#                 negative += 1
#             else:
#                 attributes.update(label="true")
#                 positive += 1
#         else:
#             attributes.update(label="true")
#             positive += 1
#
#         labelled_case._set_attributes(attributes)
#         labelled_log.append(labelled_case)
#     print('positive: ', str(positive), '; negative: ', str(negative))
#
#     return labelled_log

#
# train 2 3rd level
#
# def label_log(self, log):
#     labelled_log = lg.EventLog()
#     self.attributes = log._attributes
#     positive = 0
#     negative = 0
#     for case_index, case in enumerate(log):
#         labelled_case = case
#         first_event = case[0]
#         attributes = labelled_case._get_attributes()
#         if int(first_event["Age"]) >= 45:
#             if first_event["CType"] == 'Regular' or first_event["CType"] == 'Silver':
#                 if first_event["ClType"] == 'H11' or first_event["ClType"] == 'H12' or first_event["ClType"] == 'H13':
#                     attributes.update(label="true")
#                     positive += 1
#                 else:
#                     attributes.update(label="false")
#                     negative += 1
#             else:
#                 attributes.update(label="false")
#                 negative += 1
#         else:
#             attributes.update(label="false")
#             negative += 1
#         labelled_case._set_attributes(attributes)
#         labelled_log.append(labelled_case)
#     print('positive: ', str(positive), '; negative: ', str(negative))
#
#     return labelled_log
