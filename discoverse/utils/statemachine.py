class SimpleStateMachine:
    state_idx = 0
    state_cnt = 0
    new_state = True
    max_state_cnt = -1

    def next(self):
        if self.state_idx < self.max_state_cnt:
            self.state_cnt = 0
            self.new_state = True
            self.state_idx += 1

    def trigger(self):
        if self.new_state:
            self.new_state = False
            return True
        else:
            return False

    def update(self):
        self.state_cnt += 1

    def reset(self):
        self.state_idx = 0
        self.state_cnt = 0
        self.new_state = True
