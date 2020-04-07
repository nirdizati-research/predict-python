"""
Testing functions used in deviance mining process and that all the templates methods work as intended
"""


import unittest
from src.encoding.declare.declaretemplates import *
from src.encoding.declare.declarecommon import *

config_2 = {
    "label" : "Label",
    "deviant" : str(1),
    "nondeviant" : str(0),
    "type" : "string",
    "shuffle" : False
}


def el_to_pos_events(event_list):

    events = defaultdict(list)
    for pos, event in enumerate(event_list):
        # transition? not for now
        events[event].append(pos)

    return events

def el_to_pos_events_list(event_lists):

    lst = []

    for event_list in event_lists:
        events = defaultdict(list)
        for pos, event in enumerate(event_list):
            # transition? not for now
            events[event].append(pos)
        lst.append(events)

    return lst



def split_to_list(event_lists):
    """
    like unit split, but input has True, False as well
    :param event_lists:
    :return:
    """
    lists = []
    for event_list, deviant in event_lists:
        lists.append((event_list.split("-"), deviant))

    return lists


def unit_split(trace):
    """
    Takes trace in form of["A-B-C", "A"], splits into list [["A","B","C"], ["A"]]
    :param trace:
    :return:
    """
    lists = []
    for event_list in trace:
        lists.append(event_list.split("-"))

    return lists


class TestDeclareTemplates(unittest.TestCase):
    def test_init(self):
        """
        2 traces, one with init first, second with not
        Tests if init template works correctly
        """

        traces = [
            "A-B", # dev
            "B-A" # nondev
        ]
        event_lists = unit_split(traces)
        pos0 = el_to_pos_events(event_lists[0])
        pos1 = el_to_pos_events(event_lists[1])

        res, _ = template_init(pos0, ("A",))
        res2, _ = template_init(pos1, ("A",))

        self.assertEqual(res, 1)
        self.assertEqual(res2, -1)


    def test_absence(self):
        """
        Tests absence template
        :return:
        """

        traces = [
            "A-B",
            "B",
            "B-A"
        ]

        event_lists = unit_split(traces)

        pos = el_to_pos_events_list(event_lists)

        not_abs_1, _ = template_absence1(pos[0], ("A", ))

    def test_exactly(self):
        """
        Three traces:
        no event, 1 event, 2 event, 3 event, 4 event.
        :return:
        """

        traces = [
            "B", # 0
            "A-B", # 1
            "A-A-B", # 2
            "A-A-A-B", # 3
        ]
        event_lists = unit_split(traces)

        pos = el_to_pos_events_list(event_lists)



        zero, _ = template_exactly1(pos[0], ("A",))
        one, _ = template_exactly1(pos[1], ("A",))
        two, _ = template_exactly2(pos[2], ("A",))
        three, _ = template_exactly3(pos[3], ("A",))
        two_f, _ = template_exactly2(pos[3], ("A"))
        three_f, _ = template_exactly3(pos[2], ("A"))

        self.assertEqual(zero, -1)
        self.assertEqual(one, 1)
        self.assertEqual(two, 1)
        self.assertEqual(three, 1)
        self.assertEqual(two_f, 0)
        self.assertEqual(three_f, 0)



    def test_existence(self):
        """
        Three traces:
        no event, 1 event, 2 event, 3 event, 4 event.
        :return:
        """

        traces = [
            "B", # 0
            "A-B", # 1
            "A-A-B", # 2
            "A-A-A-B", # 3
        ]
        event_lists = unit_split(traces)

        pos = el_to_pos_events_list(event_lists)



        zero, _ = template_exist(pos[0], ("A",))
        one, _ = template_exist(pos[1], ("A",))
        two, _ = template_exist(pos[2], ("A",))

        self.assertEqual(zero, -1)
        self.assertEqual(one, 1)
        self.assertEqual(two, 2)




    def test_choice(self):
        """
=       Only one of two events exist and not both.
        :return:
        """

        traces = [
            "B-C-A", # 0
            "A-C-D", # 1
            "B-D-C", # 2
            "D-C"
        ]
        event_lists = unit_split(traces)

        pos = el_to_pos_events_list(event_lists)



        zero, _ = template_choice(pos[0], ("A","B"))
        one, _ = template_choice(pos[1], ("A","B"))
        two, _ = template_choice(pos[2], ("A","B"))
        vac, _ = template_choice(pos[3], ("A","B"))


        self.assertEqual(zero, -1)
        self.assertEqual(one, 1)
        self.assertEqual(two, 1)
        self.assertEqual(vac, 0)



    def test_coexistence(self):
        """
        Only one of two events exist and not both.
        :return:
        """

        traces = [
            "B-C-A", # 0
            "A-C-D", # 1
            "B-D-C", # 2
            "D-C"
        ]
        event_lists = unit_split(traces)

        pos = el_to_pos_events_list(event_lists)


        zero, _ = template_coexistence(pos[0], ("A","B"))
        one, _ = template_coexistence(pos[1], ("A","B"))
        two, _ = template_coexistence(pos[2], ("A","B"))
        vac, t = template_coexistence(pos[3], ("A","B"))


        self.assertEqual(zero, 1)
        self.assertEqual(one, -1)
        self.assertEqual(two, 0)
        self.assertEqual(vac, 1)
        self.assertEqual(t, True) # With vacuity check..



    def test_alternate_precedence(self):
        """
        If B, then it must be preceded by A, and before A (backwards), there cant be any more B's

        :return:
        """
        """
        Every B is preceded by A.
        """
        traces = [
            "A-B",  # true
            "B",  # false
            "A-B-A-B-A-A-A-B",  # true
            "A-B-A-B-B-A-A" #false
        ]
        event_lists = unit_split(traces)

        pos = el_to_pos_events_list(event_lists)

        zero, _ = template_alternate_precedence(pos[0], ("A", "B"))
        one, _ = template_alternate_precedence(pos[1], ("A", "B"))
        two, _ = template_alternate_precedence(pos[2], ("A", "B"))
        vac, t = template_alternate_precedence(pos[3], ("A", "B"))
        vac2, t = template_alternate_precedence(pos[0], ("B", "A"))

        self.assertEqual(zero, 1)
        self.assertEqual(one, -1)
        self.assertEqual(two, 3)
        self.assertEqual(vac, 0)
        self.assertEqual(vac2, 0)

    def test_alternate_response(self):
        """
        If A, then it must be followed by B, meanwhile there cant be any extra A

        :return:
        """
        """
        Every B is preceded by A.
        """
        traces = [
            "A-B",  # true
            "B",  # false
            "A-B-A-B-A-A-A-B",  # false
            "A-B-A-B-B-A-B-B" # true
        ]
        event_lists = unit_split(traces)

        pos = el_to_pos_events_list(event_lists)

        zero, _ = template_alternate_response(pos[0], ("A", "B"))
        one, vact1 = template_alternate_response(pos[1], ("A", "B"))
        two, _ = template_alternate_response(pos[2], ("A", "B"))
        vac, _ = template_alternate_response(pos[3], ("A", "B"))
        vac2, _ = template_alternate_response(pos[0], ("B", "A"))

        self.assertEqual(zero, 1)
        self.assertEqual(one, 0) # 1 and 0 by vacuity
        self.assertEqual(vact1, True) # 1 and 0 by vacuity
        self.assertEqual(two, 0) # too many A in row
        self.assertEqual(vac, 3) # true, no 2 A in row
        self.assertEqual(vac2, 0)  # no response


    def test_alternate_succession(self):
        """
        If B, then it must be preceded by A, and before A (backwards), there cant be any more B's

        :return:
        """
        """
        Every B is preceded by A.
        """
        traces = [
            "A-B",  # true
            "B",  # false
            "A-B-A-B-A-A-A-B",  # false
            "A-B-A-B-B-A-B-B-A-A", # false,
            "A-B-C-A-B-D-D-D-A-B-D-A-C-B" # true
        ]
        event_lists = unit_split(traces)

        pos = el_to_pos_events_list(event_lists)

        zero, _ = template_alternate_succession(pos[0], ("A", "B"))
        one, vact1 = template_alternate_succession(pos[1], ("A", "B"))
        two, _ = template_alternate_succession(pos[2], ("A", "B"))
        vac, _ = template_alternate_succession(pos[3], ("A", "B"))
        vac2, _ = template_alternate_succession(pos[4], ("A", "B"))
        vac3, _ = template_alternate_succession(pos[4], ("B", "A"))

        self.assertEqual(zero, 1) # true
        self.assertEqual(one, -1) # false
        self.assertEqual(two, 0) # too many A in row
        self.assertEqual(vac, 0) # too many B in a row
        self.assertEqual(vac2, 4)  # true, 4
        self.assertEqual(vac3, 0)  # false



    def test_chain_precedence(self):
        """
        Every B must be next of an A.
        """
        traces = [
            "A-B",  # true
            "B",  # false
            "A-B-A-B-A-A-A-B",  # false
            "A-B-A-B-B-A-B-B-A-A", # false,
            "A-B-C-A-B-D-D-D-A-B-D-A-C-B", # false,
            "A-C-B",
        ]
        event_lists = unit_split(traces)

        pos = el_to_pos_events_list(event_lists)

        zero, _ = template_chain_precedence(pos[0], ("A", "B"))
        one, vact1 = template_chain_precedence(pos[1], ("A", "B"))
        two, _ = template_chain_precedence(pos[2], ("A", "B"))
        vac, _ = template_chain_precedence(pos[3], ("A", "B"))
        vac2, _ = template_chain_precedence(pos[4], ("A", "B"))
        vac3, _ = template_chain_precedence(pos[4], ("B", "A"))
        vac4, _ = template_chain_precedence(pos[5], ("A", "B"))

        self.assertEqual(zero, 1) # true
        self.assertEqual(one, -1) # false
        self.assertEqual(two, 3) #  true, 3 B's have A before it
        self.assertEqual(vac, 0) # too many B in a row
        self.assertEqual(vac2, 0)  # false, last B has C instead of A before it
        self.assertEqual(vac3, 0)  # false
        self.assertEqual(vac4, 0)  # false, C before B


    def test_chain_response(self):
        """
        Every A must be next straight followed by B.
        """
        traces = [
            "A-B",  # true
            "B",  # true, false by vacuity
            "A-B-A-B-A-A-A-B",  # false
            "A-B-A-B-B-A-B-B-A-A", # false,
            "A-B-C-A-B-D-D-D-A-B-D-A-B", # true,
            "A-C-B", # false
        ]
        event_lists = unit_split(traces)

        pos = el_to_pos_events_list(event_lists)

        zero, _ = template_chain_response(pos[0], ("A", "B"))
        one, vact1 = template_chain_response(pos[1], ("A", "B"))
        vac, _ = template_chain_response(pos[3], ("A", "B"))
        vac2, _ = template_chain_response(pos[4], ("A", "B"))
        vac3, _ = template_chain_response(pos[4], ("B", "A"))
        vac4, _ = template_chain_response(pos[5], ("A", "B"))

        self.assertEqual(zero, 1) # true
        self.assertEqual(one, 0) # true by vacuity
        self.assertEqual(vact1, True) # true
        self.assertEqual(vac, 0) # Last A's not followed by B
        self.assertEqual(vac2, 4)  # true
        self.assertEqual(vac3, 0)  # false
        self.assertEqual(vac4, 0)  # false, A not straight followed by B


    def test_chain_succession(self):
        """
        Every A must be next straight followed by B
        Every B must be instantly preceded by A
        """
        traces = [
            "C-D", # true by vacuity
            "A-B",  # true
            "B",  # false
            "A-B-A-B-A-B-C-D-C-A-B-A-B",  # true
            "A-B-A-B-B-A-B-B-A-A", # false,
            "A-B-C-A-B-D-D-D-A-B-D-A-B", # true,
            "A-C-B", # false
        ]
        event_lists = unit_split(traces)

        pos = el_to_pos_events_list(event_lists)

        zero, vact1 = template_chain_succession(pos[0], ("A", "B"))
        one, _ = template_chain_succession(pos[1], ("A", "B"))
        vac, _ = template_chain_succession(pos[2], ("A", "B"))
        vac2, _ = template_chain_succession(pos[3], ("A", "B"))
        vac3, _ = template_chain_succession(pos[4], ("A", "B"))
        vac4, _ = template_chain_succession(pos[5], ("A", "B"))
        vac5, _ = template_chain_succession(pos[6], ("A", "B"))

        self.assertEqual(zero, 0) # true by vacuity
        self.assertEqual(vact1, True) # true
        self.assertEqual(one, 1) # true
        self.assertEqual(vac, 0) # false, B alone
        self.assertEqual(vac2, 5)  # true
        self.assertEqual(vac3, 0)  # false
        self.assertEqual(vac4, 4)  # true
        self.assertEqual(vac5, 0) # false, A not straight followed by B



    def test_not_chain_succession(self):
        """
        B must not be preceded by A and A must not be followed by B,
        B-A is ok, A-B is not
        :return:
        """

        traces = [
            "C-D", # true by vacuity
            "A-B",  # false
            "B",  # true, true by vacuity?
            "A-B-A-B-A-B-C-D-C-A-B-A-B",  # false
            "A-B-A-B-B-A-B-B-A-A", # false,
            "B-A-C-B-A-D-D-D-B-A-D-B-A", # true,
            "A-C-B", # true
        ]
        event_lists = unit_split(traces)

        pos = el_to_pos_events_list(event_lists)

        zero, vact1 = template_not_chain_succession(pos[0], ("A", "B"))
        one, _ = template_not_chain_succession(pos[1], ("A", "B"))
        vac, _ = template_not_chain_succession(pos[2], ("A", "B"))
        vac2, _ = template_not_chain_succession(pos[3], ("A", "B"))
        vac3, _ = template_not_chain_succession(pos[4], ("A", "B"))
        vac4, _ = template_not_chain_succession(pos[5], ("A", "B"))
        vac5, _ = template_not_chain_succession(pos[6], ("A", "B"))

        self.assertEqual(zero, 0) # true by vacuity
        self.assertEqual(vact1, True) # true
        self.assertEqual(one, 0) # false
        self.assertEqual(vac, 1) # false, B alone, not sure on vacuity..
        self.assertEqual(vac2, 0)  # false
        self.assertEqual(vac3, 0)  # false
        self.assertEqual(vac4, 1)  # true
        self.assertEqual(vac5, 1) # true, A not straight followed by B


    def test_not_coexistence(self):
        """
        A and B must not exist together in same trace
        """
        traces = [
            "C-D",  # true by vacuity
            "A-B",  # false
            "B",  # true
            "A-C-A-C-A-G-C-C-C-F-G-A-G",  # true
            "A-B-A-B-B-A-B-B-A-A",  # false,
            "B-A-C-B-A-D-D-D-B-A-D-B-A",  # false,
            "B-C",  # true
        ]
        event_lists = unit_split(traces)

        pos = el_to_pos_events_list(event_lists)

        zero, vact1 = template_not_coexistence(pos[0], ("A", "B"))
        one, _ = template_not_coexistence(pos[1], ("A", "B"))
        vac, _ = template_not_coexistence(pos[2], ("A", "B"))
        vac2, _ = template_not_coexistence(pos[3], ("A", "B"))
        vac3, _ = template_not_coexistence(pos[4], ("A", "B"))
        vac4, _ = template_not_coexistence(pos[5], ("A", "B"))
        vac5, _ = template_not_coexistence(pos[6], ("A", "B"))

        self.assertEqual(zero, 0)  # true by vacuity
        self.assertEqual(vact1, True)  # true
        self.assertEqual(one, 0)  # false
        self.assertEqual(vac, 1)  # true
        self.assertEqual(vac2, 1)  # true
        self.assertEqual(vac3, 0)  # false
        self.assertEqual(vac4, 0)  # false
        self.assertEqual(vac5, 1)  # true, B, but not A

    def test_not_succession(self):
        """
        A must not eventually be followed by B, B-C-A is ok, A-B-C is not
        """

        traces = [
            "C-D",  # true by vacuity
            "A-B",  # false
            "B",  # true by vacuity, yes or no?
            "A-C-A-C-A-G-C-C-C-F-G-A-G",  # true, no B at all, is it vac or not_
            "A-B-A-B-B-A-B-B-A-A",  # false, a followed by b
            "B-A-C-B-A-D-D-D-B-A-D-B-A",  # false
            "A-C-D-E-F-G-B" # false
            ]

        event_lists = unit_split(traces)


        pos = el_to_pos_events_list(event_lists)

        zero, vact1 = template_not_succession(pos[0], ("A", "B"))
        one,  _ = template_not_succession(pos[1], ("A", "B"))
        vac, vact2 = template_not_succession(pos[2], ("A", "B"))
        vac2, _ = template_not_succession(pos[3], ("A", "B"))
        vac3, _ = template_not_succession(pos[4], ("A", "B"))
        vac4, _ = template_not_succession(pos[5], ("A", "B"))
        vac5, _ = template_not_succession(pos[6], ("A", "B"))

        self.assertEqual(zero, 0)  # true by vacuity
        self.assertEqual(vact1, True)  # not sure
        self.assertEqual(one, 0)  # A followed by B, false
        self.assertEqual(vact2, True)  # true not sure..
        self.assertEqual(vac, 1)  # true
        self.assertEqual(vac2, 1)  # true
        self.assertEqual(vac3, 0)  # false
        self.assertEqual(vac4, 0)  # false
        self.assertEqual(vac5, 0)  # false, A eventually followed by B

    def test_precedence(self):
        """
        B must be preceded by A
        :return:
        """


        traces = [
            "C-D",  # true by vacuity
            "A-B",  # true
            "B",  # false, no A before
            "A-C-A-C-A-G-C-C-C-F-G-A-G",  # True by Vacuity
            "A-B-A-B-B-A-B-B-A-A",  # true, exists B where it is preceded by A
            "B-A-C-B-A-D-D-D-B-A-D-B-A",  # false, first B is not preceded
            "A-C-D-E-F-G-B"  # true
        ]

        event_lists = unit_split(traces)

        pos = el_to_pos_events_list(event_lists)

        zero, vact1 = template_precedence(pos[0], ("A", "B"))
        one, _ = template_precedence(pos[1], ("A", "B"))
        vac, _ = template_precedence(pos[2], ("A", "B"))
        vac2, vact2 = template_precedence(pos[3], ("A", "B"))
        vac3, _ = template_precedence(pos[4], ("A", "B"))
        vac4, _ = template_precedence(pos[5], ("A", "B"))
        vac5, _ = template_precedence(pos[6], ("A", "B"))

        self.assertEqual(zero, 0)  # true, no B at all
        self.assertTrue(vact1)
        self.assertEqual(one, 1)  #
        self.assertEqual(vac, 0)
        self.assertEqual(vac2, 1)
        self.assertTrue(vact2)
        self.assertEqual(vac3, 5)
        self.assertEqual(vac4, 0)
        self.assertEqual(vac5, 1)

    def test_response(self):
        """
        A must be followed by B
        :return:
        """


        traces = [
            "C-D",  # true by vacuity
            "A-B",  # true
            "B",  # true by vacuity
            "A-C-A-C-A-G-C-C-C-F-G-A-G",  # false
            "A-B-A-B-B-A-B-B-A-A",  # false, last A is not followed by B
            "B-A-C-B-A-D-D-D-B-A-D-B",  # true, every A followed by B
            "A-C-D-E-F-G-B"  # true
        ]

        event_lists = unit_split(traces)

        pos = el_to_pos_events_list(event_lists)

        zero, vact1 = template_response(pos[0], ("A", "B"))
        one, _ = template_response(pos[1], ("A", "B"))
        two, vact2 = template_response(pos[2], ("A", "B"))
        three, _ = template_response(pos[3], ("A", "B"))
        four, _ = template_response(pos[4], ("A", "B"))
        five, _ = template_response(pos[5], ("A", "B"))
        six, _ = template_response(pos[6], ("A", "B"))

        self.assertEqual(zero, 0)  # true by vacuity
        self.assertTrue(vact1)
        self.assertEqual(one, 1)  # A followed by
        self.assertEqual(two, 1)  # true vacuity
        self.assertTrue(vact2)
        self.assertEqual(three, 0)  # false
        self.assertEqual(four, 0) # fasle
        self.assertEqual(five, 3)  # true
        self.assertEqual(six, 1)  # true


    def test_responded_existence(self):
        """
        If A exists, then B must exist. Other way around doesnt have to be ture
        :return:
        """
        traces = [
            "C-D",  # true by vacuity
            "A-B",  # true
            "B",  # true by vacuity
            "A-C-A-C-A-G-C-C-C-F-G-A-G",  # false
            "A-B-A-B-B-A-B-B-A-A",  # true
            "B-A-C-B-A-D-D-D-B-A-D-B",  # true
            "A-C-D-E-F-G-B"  # true
        ]

        event_lists = unit_split(traces)

        pos = el_to_pos_events_list(event_lists)

        zero, vact1 = template_responded_existence(pos[0], ("A", "B"))
        one, _ = template_responded_existence(pos[1], ("A", "B"))
        two, vact2 = template_responded_existence(pos[2], ("A", "B"))
        three, _ = template_responded_existence(pos[3], ("A", "B"))
        four, _ = template_responded_existence(pos[4], ("A", "B"))
        five, _ = template_responded_existence(pos[5], ("A", "B"))
        six, _ = template_responded_existence(pos[6], ("A", "B"))

        self.assertEqual(zero, 0)  # true by vacuity
        self.assertTrue(vact1)
        self.assertEqual(one, 1)  # A followed by
        self.assertEqual(two, 1)  # true vacuity
        self.assertTrue(vact2)
        self.assertEqual(three, 0)  # false
        self.assertEqual(four, 5) # true
        self.assertEqual(five, 3)  # true
        self.assertEqual(six, 1)  # true

    def test_succession(self):
        """
        If A exists, it must be followed by B, if B exists it must be followed by A
        :return:
        """

        traces = [
            "C-D",  # true by vacuity
            "A-B",  # true
            "B",  # false
            "A-C-A-C-A-G-C-C-C-F-G-A-G",  # false
            "A-B-A-B-B-A-B-B-A-A",  # false
            "B-A-C-B-A-D-D-D-B-A-D-B",  # false
            "A-C-D-E-F-G-B-A-B"  # true
        ]

        event_lists = unit_split(traces)

        pos = el_to_pos_events_list(event_lists)

        zero, vact1 = template_succession(pos[0], ("A", "B"))
        one, _ = template_succession(pos[1], ("A", "B"))
        two, vact2 = template_succession(pos[2], ("A", "B"))
        three, _ = template_succession(pos[3], ("A", "B"))
        four, _ = template_succession(pos[4], ("A", "B"))
        five, _ = template_succession(pos[5], ("A", "B"))
        six, _ = template_succession(pos[6], ("A", "B"))

        self.assertEqual(zero, 1)  # true by vacuity
        self.assertFalse(vact1)
        self.assertEqual(one, 1)  # A followed by
        self.assertEqual(two, 0)  # false
        self.assertEqual(three, 0)  # false
        self.assertEqual(four, 0) # false
        self.assertEqual(five, 0)  # false
        self.assertEqual(six, 2)  # true



"""
templates:

"""



"""
def test_thorough():
    filepath = "tests/"+ "test_exactly.xes"
    lg = LogGenerator(config_2)

    traces = [
        ("B", True),
        ("A-B", False),
        ("B-A-A", False)
    ]
    event_lists = split_to_list(traces)

    lg.create_from_event_lists(event_lists)

    xes = lg.convert_to_xes()
    open(filepath, "w").write(str(xes))
    log = read_XES_log(filepath)
    positional = xes_to_positional(log)

    res, _ = template_init(positional[0]["events"], ("A",))
    res2, _ = template_init(positional[1]["events"], ("A",))

    assert (res == 1)
    assert (res2 == 0)
"""





if __name__ == "__main__":
    unittest.main()
