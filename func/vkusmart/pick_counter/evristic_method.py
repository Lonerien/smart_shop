from collections import defaultdict

class EvristicMethod:
    def __init__(self,
                streak,
                good_in_streak,
                initial_timestamp,
                time_step
                ):
        self.hand_balance = defaultdict(int)  # left_state/right_state ---> balance
        self.hand_balance['left_state'] = 0
        self.hand_balance['right_state'] = 0
        self.streak = streak
        self.good_in_streak = good_in_streak
        self.already_counted_pick = {"left_state": False, "right_state": False}
        self.already_put_back = {"left_state": False, "right_state": False}
        self.initial_timestamp = initial_timestamp
        self.time_step = time_step
        
    def predict(self, cur_time, arm_state_history, is_move_from_shelf_to_body):
        '''
        self.streak = инициализируется в pick_counter
        self.good_in_streak = 
        '''
        
        
#         print('self.streak=', self.streak)
#         print('self.good_in_strea=', self.good_in_streak)
        
        
        is_picked = {"left_state": 0, "right_state": 0}
        is_grep = {"left_state": False, "right_state": False}
        # Step one: make desicion
        for arm_state_type in ['left_state', 'right_state']:

            if arm_state_history[cur_time][arm_state_type] == 1:
                self.hand_balance[arm_state_type] += 1

            history_len = int((cur_time - self.initial_timestamp) / self.time_step) + 1

            if history_len >= self.streak and arm_state_history[cur_time - self.time_step * self.streak][
                arm_state_type] == 1:
                self.hand_balance[arm_state_type] -= 1 if self.hand_balance[arm_state_type] > 0 else 0

            if self.hand_balance[arm_state_type] >= self.good_in_streak:
                is_grep[arm_state_type] = True  # grep
                continue

        # Step two: consider motion direction
        is_try_take = {"left_state": None, "right_state": None}
        for arm_state_type in ['left_state', 'right_state']:
            is_try_take[arm_state_type] = is_move_from_shelf_to_body[arm_state_type]  # True -> take, False -> put back

        # Step tree: make conclusion:
        for arm_state_type in ['left_state', 'right_state']:
            if is_try_take[arm_state_type] and is_grep[arm_state_type]:
                if not self.already_counted_pick[arm_state_type]:
                    is_picked[arm_state_type] = 1
                    self.already_counted_pick[arm_state_type] = True

            if is_try_take[arm_state_type] == False and is_grep[arm_state_type]:
                if not self.already_put_back[arm_state_type]:
                    is_picked[arm_state_type] = -1
                    self.already_put_back[arm_state_type] = True

        # Step four: эвристика, считающая что товар уже взят, и выполняется другое(следующее) действие
        # пусть бейзлайн будет нереально простым
        for arm_state_type in ['left_state', 'right_state']:
            if is_try_take[arm_state_type]:
                self.already_put_back[arm_state_type] = False
            if is_try_take[arm_state_type] == False:
                self.already_counted_pick[arm_state_type] = False

        return is_picked