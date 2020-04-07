"""
Second version of declare templates.
In this case, if constraint is violated, then returns -1.
"""

from typing import Tuple


def template_absence1(trace, event_set):
    assert (len(event_set) == 1)
    event = event_set[0]
    if event in trace:
        return -1, False

    return 1, False


def template_absence2(trace, event_set):
    assert (len(event_set) == 1)
    event = event_set[0]

    if event in trace and len(trace[event]) > 1:
        return -1, False

    return 1, False


def template_absence3(trace, event_set):
    assert (len(event_set) == 1)
    event = event_set[0]

    if event in trace and len(trace[event]) > 2:
        return -1, False

    return 1, False


def template_init(trace, event_set):
    # If event is in the first position
    assert (len(event_set) == 1)

    event = event_set[0]

    if event in trace and trace[event][0] == 0:
        return 1, False

    return -1, False


def template_exist(trace, event_set):
    # if event exists in trace
    assert (len(event_set) == 1)
    event = event_set[0]

    if event in trace:
        return len(trace[event]), False

    return -1, False


def template_choice(trace, event_set):
    # at least one, but not both XOR operation

    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    if (event_1 in trace) != (event_2 in trace):
        if event_1 in trace:
            return len(trace[event_1]), False
        else:
            return len(trace[event_2]), False

    return -1, False


def template_coexistence(trace, event_set):
    # both must exist or not exist at the same time

    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    if event_1 in trace and event_2 in trace:
        # return minimum of both existence count
        return min(len(trace[event_1]), len(trace[event_2])), False

    elif event_1 not in trace and event_2 not in trace:
        return 0, True

    # Only one exists, violation
    return -1, False


def template_alternate_precedence(trace, event_set):
    """
      precedence(A, B) template indicates that event B
      should occur only if event A has occurred before.

      Alternate condition:
      "events must alternate without repetitions of these events in between"

      :param trace:
      :param event_set:
      :return:
      """

    # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]
    if event_2 in trace:
        if event_1 in trace:
            # Go through two lists, one by one
            # first events pos must be before 2nd lists first pos etc...
            # A -> A -> B -> A -> B

            # efficiency check
            event_1_count = len(trace[event_1])
            event_2_count = len(trace[event_2])

            # There has to be more or same amount of event A's compared to B's
            if event_2_count > event_1_count:
                return 0, False

            event_1_positions = trace[event_1]
            event_2_positions = trace[event_2]

            # Go through all event 2's, check that there is respective event 1.
            # Find largest event 1 position, which is smaller than event 2 position

            # implementation
            # Check 1-forward, the 1-forward has to be greater than event 2 and current one has to be smaller than event2

            event_1_ind = 0
            for i, pos2 in enumerate(event_2_positions):
                # find first in event_2_positions, it has to be before next in event_1_positions

                while True:
                    if event_1_ind >= len(event_1_positions):
                        # out of preceding events, but there are still event 2's remaining.
                        return -1, False

                    next_event_1_pos = None

                    if event_1_ind < len(event_1_positions) - 1:
                        next_event_1_pos = event_1_positions[event_1_ind + 1]

                    event_1_pos = event_1_positions[event_1_ind]

                    if next_event_1_pos:
                        if event_1_pos < pos2 and next_event_1_pos > pos2:
                            # found the largest preceding event
                            event_1_ind += 1
                            break
                        elif event_1_pos > pos2 and next_event_1_pos > pos2:
                            # no event larger
                            return -1, False
                        else:
                            event_1_ind += 1


                    else:
                        # if no next event, check if current is smaller
                        if event_1_pos < pos2:
                            event_1_ind += 1
                            break
                        else:
                            return -1, False  # since there is no smaller remaining event

            count = len(event_2_positions)
            return count, False


        else:
            # impossible because there has to be at least one event1 with event2
            return -1, False

    return 0, True  # todo: vacuity condition!!


def template_alternate_response(trace, event_set):
    """
    If there is A, it has to be eventually followed by B.
    Alternate: there cant be any further A until first next B
    :param trace:
    :param event_set:
    :return:
    """
    # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    if event_1 in trace:
        if event_2 in trace:

            event_2_ind = 0

            event_1_positions = trace[event_1]
            event_2_positions = trace[event_2]

            for i, pos1 in enumerate(event_1_positions):
                # find first in event_2_positions, it has to be before next in event_1_positions
                next_event_1_pos = None
                if i < len(event_1_positions) - 1:
                    next_event_1_pos = event_1_positions[i + 1]

                while True:
                    if event_2_ind >= len(event_2_positions):
                        # out of response events
                        return -1, False

                    if event_2_positions[event_2_ind] > pos1:
                        # found first greater than event 1 pos
                        # check if it is smaller than next event 1
                        if next_event_1_pos and event_2_positions[event_2_ind] > next_event_1_pos:
                            # next event 2 is after next event 1..
                            return -1, False
                        else:
                            # consume event 2 and break out to next event 1
                            event_2_ind += 1
                            break

                    event_2_ind += 1

            count = len(event_1_positions)
            return count, False
            # every event 2 position has to be after respective event 1 position and before next event 2 position



        else:
            return -1, False

    # Vacuously
    return 0, True


def template_alternate_succession(trace, event_set):
    """
    A-B-A-B - ... always in pair
    # TODO: is zipping and just checking respective one okay? Not likely?
    :param trace:
    :param event_set:
    :return:
    """

    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    if (event_1 in trace) != (event_2 in trace):
        return -1, False

    if event_1 in trace and event_2 in trace:
        event_1_positions = trace[event_1]
        event_2_positions = trace[event_2]

        if len(event_1_positions) != len(event_2_positions):
            return -1, False  # impossible if not same length

        pos = -1
        current_ind = 0
        switch = False
        while current_ind < len(event_1_positions):

            # Use switch to know from which array to get next..
            if switch:
                next_pos = event_2_positions[current_ind]
                current_ind += 1
            else:
                next_pos = event_1_positions[current_ind]

            if next_pos <= pos:
                return -1, False  # next one is smaller than current

            pos = next_pos  # go to next one.
            switch = not switch  # swap array

        count = len(event_1_positions)
        return count, False

    return 0, True  # vacuity condition


def template_chain_precedence(trace, event_set):  # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    if event_2 in trace:
        if event_1 in trace:
            # Each event1 must instantly be followed by event2
            event_1_positions = trace[event_1]
            event_2_positions = trace[event_2]

            if len(event_1_positions) < len(event_2_positions):
                return -1, False  # impossible to fulfill

            event_1_ind = 0

            for i, pos2 in enumerate(event_2_positions):
                # find first event 2 which is after each event 1
                while True:
                    if event_1_ind >= len(event_1_positions):
                        return -1, False  # not enough response

                    if pos2 < event_1_positions[event_1_ind]:
                        return -1, False  # passed, no event before pos2

                    if pos2 - 1 == event_1_positions[event_1_ind]:
                        event_1_ind += 1
                        break  # found right one! Move to next B event

                    event_1_ind += 1

            count = len(event_2_positions)
            return count, False
        else:
            return -1, False  # no response for event1

    return 0, True  # todo, vacuity


def template_chain_response(trace, event_set):
    # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    if event_1 in trace:
        if event_2 in trace:
            # Each event1 must instantly be followed by event2
            event_1_positions = trace[event_1]
            event_2_positions = trace[event_2]

            if len(event_1_positions) > len(event_2_positions):
                return -1, False  # impossible to fulfill

            event_2_ind = 0

            for i, pos1 in enumerate(event_1_positions):
                # find first event 2 which is after each event 1
                while True:
                    if event_2_ind >= len(event_2_positions):
                        return -1, False  # not enough response

                    if pos1 < event_2_positions[event_2_ind]:
                        if pos1 + 1 != event_2_positions[event_2_ind]:
                            return -1, False  # next one is not straight after
                        else:
                            event_2_ind += 1
                            break  # next one is straight after move to next event1
                    event_2_ind += 1

            count = len(event_1_positions), False
            return count

        else:
            return -1, False  # no response for event1

    return 0, True  # todo, vacuity


def template_chain_succession(trace, event_set):
    """
    Everytime there is A, it has to be instantly followed by B, everytime there is B
    it has to be preceded by A
    :param trace:
    :param event_set:
    :return:
    """

    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    if (event_1 in trace) != (event_2 in trace):
        return -1, False

    if event_1 in trace and event_2 in trace:
        event_1_positions = trace[event_1]
        event_2_positions = trace[event_2]

        if len(event_1_positions) != len(event_2_positions):
            # has to be same number of events
            return -1, False

        # They have to appear together, with event1 always before event2
        for i in range(len(event_1_positions)):
            if event_1_positions[i] + 1 != event_2_positions[i]:
                return -1, False

        count = len(event_1_positions)
        return count, False

    return 0, True  # todo vacuity


def template_exactly1(trace, event_set):
    # exactly 1 event

    event = event_set[0]

    if event in trace and len(trace[event]) == 1:
        return 1, False

    return -1, False


def template_exactly2(trace, event_set):
    # exactly 2 events
    assert (len(event_set) == 1)

    event = event_set[0]

    if event in trace and len(trace[event]) == 2:
        return 1, False

    return -1, False


def template_exactly3(trace, event_set):
    # exactly 3 event
    assert (len(event_set) == 1)

    event = event_set[0]

    if event in trace and len(trace[event]) == 3:
        return 1, False

    return -1, False


def template_not_chain_succession(trace, event_set):
    """
    TODO: check vacuity conditions for not templates.
    :param trace:
    :param event_set:
    :return:
    """
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    if event_1 in trace and event_2 in trace:
        # Find a place, where A and B are next
        event_1_positions = trace[event_1]
        event_2_positions = trace[event_2]

        e1_ind = 0
        e2_ind = 0
        while True:
            if e1_ind >= len(event_1_positions) or e2_ind >= len(event_2_positions):
                return 1, False  # no more choices

            current_e1 = event_1_positions[e1_ind]
            current_e2 = event_2_positions[e2_ind]

            if current_e1 > current_e2:
                e2_ind += 1
            else:
                if current_e1 + 1 == current_e2:
                    return -1, False  # found a place, where they are together
                e1_ind += 1

    # How to do vacuity here? 1 by default most likely
    return 0, True  # TODO, this condition?


def template_not_coexistence(trace, event_set):
    # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    if event_1 in trace and event_2 in trace:
        return -1, False  # if both in trace, then they exist together.
    elif event_1 in trace or event_2 in trace: # only one exists in trace
        return 1, False

    # if neither in trace, vacuously fulfilled
    return 0, True


def template_not_succession(trace, event_set):
    # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    if event_1 in trace:
        if event_2 in trace:

            # for this to be true, last event 2 has to be before first event 1
            first_event_1 = trace[event_1][0]
            last_event_2 = trace[event_2][-1]

            if first_event_1 < last_event_2:
                return -1, False  # in this case there is an event 2, which appears after first event
            else:
                return 1, False
        else:
            return 1, False  # not possible

    # if not, then impossible and template fulfilled
    return 0, True  # vacuity


def template_precedence(trace, event_set):
    """
    precedence(A, B) template indicates that event B
    should occur only if event A has occurred before.
    :param trace:
    :param event_set:
    :return:
    """

    # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    if event_2 in trace:
        if event_1 in trace:
            first_pos_event_1 = trace[event_1][0]
            first_pos_event_2 = trace[event_2][0]
            if first_pos_event_1 < first_pos_event_2:
                # todo: check frequency condition
                count = min(len(trace[event_1]), len(trace[event_2]))
                return count, False
            else:
                # first position of event 2 is before first event 1
                return -1, False

        else:
            # impossible because there has to be at least one event1 with event2
            return -1, False

    # Vacuously fulfilled
    return 0, True


def template_response(trace, event_set):
    """
    If event B is the response of event A, then when event
    A occurs, event B should eventually occur after A.
    :param trace:
    :param event_set:
    :return:
    """
    # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    if event_1 in trace:
        if event_2 in trace:
            last_pos_event_1 = trace[event_1][-1]
            last_pos_event_2 = trace[event_2][-1]
            if last_pos_event_2 > last_pos_event_1:
                # todo: check frequency counting How to count fulfillments? min of A and B?
                count = min(len(trace[event_1]), len(trace[event_2]))
                return count, False

            else:
                # last event2 is before event1
                return -1, False
        else:
            # impossible for event 2 to be after event 1 if there is no event 2
            return -1, False

    return 0, True  # not vacuity atm..


def template_responded_existence(trace, event_set):
    """
    The responded existence(A, B) template specifies that
    if event A occurs, event B should also occur (either
        before or after event A).
    :return:
    """

    # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    if event_1 in trace:
        if event_2 in trace:
            return min(len(trace[event_1]), len(trace[event_2])), False
        else:
            return -1, False

    return 0, True  # 0, if vacuity condition


def template_succession(trace, event_set) -> Tuple[int, bool]:
    """
    succession(A, B) template requires that both response and
    precedence relations hold between the events A and B.
    :return:

    If A appears, then eventually there has to be B
    If B appears, then it has to be preceded by A
    """
    # exactly 2 event
    assert (len(event_set) == 2)

    event_1 = event_set[0]
    event_2 = event_set[1]

    if event_1 in trace and not event_2 in trace:
        return -1, False

    if event_2 in trace and not event_1 in trace:
        return -1, False

    if event_1 in trace and event_2 in trace:
        # First position of A
        first_pos_event_1 = trace[event_1][0]

        # First position of B
        first_pos_event_2 = trace[event_2][0]

        # Last position A
        last_pos_event_1 = trace[event_1][-1]

        # Last position B
        last_pos_event_2 = trace[event_2][-1]

        if first_pos_event_1 < first_pos_event_2 and last_pos_event_1 < last_pos_event_2:
            # todo: check frequency!
            count = min(len(trace[event_1]), len(trace[event_2]))
            return count, False
        else:
            return -1, False

    # todo: vacuity condition!
    return 0, True


# Does order matter in template?
template_order = {
    "choice": False,
    "coexistence": False,
    "alternate_precedence": True,
    "alternate_succession": True,
    "alternate_response": True,
    "chain_precedence": True,
    "chain_response": True,
    "chain_succession": True,
    "not_chain_succession": True,
    "not_coexistence": False,
    "not_succession": True,
    "responded_existence": True,
    "response": True,
    "succession": True,
    "precedence": True
}

template_sizes = {
                  # "init": 1,
                  "exist": 1,
                  # "absence1": 1,
                  # "absence2": 1,
                  # "absence3": 1,
                  # "choice": 2,
                  # "coexistence": 2,
                  # "exactly1": 1,
                  # "exactly2": 1,
                  # "exactly3": 1,
                  # "alternate_precedence": 2,
                  # "alternate_succession": 2,
                  # "alternate_response": 2,
                  # "chain_precedence": 2,
                  # "chain_response": 2,
                  # "chain_succession": 2,
                  # "not_chain_succession": 2,
                  # "not_coexistence": 2,
                  # "not_succession": 2,
                  # "responded_existence": 2,
                  # "response": 2,
                  # "succession": 2,
                  # "precedence": 2
                  }


def apply_template(template_str, trace, event_set):
    template_map = {
        # "init": template_init,
        "exist": template_exist,
        # "absence1": template_absence1,
        # "absence2": template_absence2,
        # "absence3": template_absence3,
        # "choice": template_choice,
        # "coexistence": template_coexistence,
        # "exactly1": template_exactly1,
        # "exactly2": template_exactly2,
        # "exactly3": template_exactly3,
        # "alternate_precedence": template_alternate_precedence,
        # "alternate_succession": template_alternate_succession,
        # "alternate_response": template_alternate_response,
        # "chain_precedence": template_chain_precedence,
        # "chain_response": template_chain_response,
        # "chain_succession": template_chain_succession,
        # "not_chain_succession": template_not_chain_succession,
        # "not_coexistence": template_not_coexistence,
        # "not_succession": template_not_succession,
        # "responded_existence": template_responded_existence,
        # "response": template_response,
        # "succession": template_succession,
        # "precedence": template_precedence
    }

    lower = template_str.lower()

    if lower in template_map:
        return template_map[lower](trace, event_set)
    else:
        raise Exception("Template not found")
