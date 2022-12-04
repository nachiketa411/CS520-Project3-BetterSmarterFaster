import copy

from Agent import Agent
from Constants import NO_OF_STEPS_4


class AgentUStar(Agent):
    def move_agent(self):
        count = 0
        while count <= NO_OF_STEPS_4:
            next_move = self.get_next_move()
            self.currPos = next_move
            self.path.append(next_move)
            if self.currPos == self.prey.currPos:
                # print("Yippiieeee")
                count += 1
                return [count, -1, self.counter_for_prey_actually_found, self.counter_for_predator_actually_found]
            elif self.currPos == self.predator.currPos:
                print("Ded")
                count += 1
                return [count, -2, self.counter_for_prey_actually_found, self.counter_for_predator_actually_found]

            self.prey.take_next_move(copy.deepcopy(self.graph))
            if self.currPos == self.prey.currPos:
                # print("Yippiieeee")
                count += 1
                return [count, -1, self.counter_for_prey_actually_found, self.counter_for_predator_actually_found]

            self.predator.take_next_move()
            if self.currPos == self.predator.currPos:
                # print("Ded")
                count += 1
                return [count, -2, self.counter_for_prey_actually_found, self.counter_for_predator_actually_found]
            count += 1
        return [count, -3, self.counter_for_prey_actually_found, self.counter_for_predator_actually_found]

    def get_next_move(self):
        return self.utility[self.currPos, self.predator.currPos, self.prey.currPos, 0]
