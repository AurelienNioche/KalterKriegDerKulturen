from pylab import plt, np
from tqdm import tqdm
from itertools import product

def set_seed(seed):
    """Initialize the seed of pseudo-random generator."""

    if seed is None:
        import time
        seed = int((time.time()*10**6) % 4294967295)
    try:
        np.random.seed(seed)
        print("Seed used for pseudo-random values generator:", seed, "\n")
    except:
        print("!!! WARNING !!!: Seed was not set correctly.")
    return seed

class Convictions(np.ndarray):

    '''
        Convictions inherits from numpy array. What it does in more is to modify the culture when he is modified himself
    '''

    def __new__(cls, array_like, culture):

        self = np.array(array_like).view(cls)
        return self

    def __init__(self, array_like, culture):

        self.parent = culture

    def __setitem__(self, instance, value):

        super().__setitem__(instance, value)
        self.parent[:] = self[:] > 0


class Culture(np.ndarray):

    '''
        Culture inherits from numpy array. What it does in more is to have a 'Convictions' object as attribute
    '''

    def __new__(cls, convictions):

        self = np.array(np.zeros(len(convictions), dtype=int)).view(cls)
        return self

    def __init__(self, convictions):

        self.convictions = Convictions(convictions, self)
        self[:] = self.convictions[:] > 0

    def contains_belief(self, belief):

        # Imagine that a belief is something like np.array([np.nan, 0, 1]) for '#01'
        return self[np.where(belief == 0)] == 0 and self[np.where(belief == 1)] == 1

    def get_most_robust_convictions(self, n=1):

        return np.argsort(np.absolute(self.convictions))[::-1][:n]


class Agent(object):

    def __init__(self, suggestibility, proselytism, convictions):

        # --- Parameters --- #
        self.suggestibility = suggestibility
        self.proselytism = proselytism
        # ----------------- #

        self.culture = Culture(convictions=convictions)

    def try_to_convince(self):

        arguments_idx = self.culture.get_most_robust_convictions(n=self.proselytism)
        arguments_strength = self.culture.convictions[arguments_idx]

        return arguments_idx, arguments_strength

    def get_influenced(self, arguments_idx, arguments_strength):

        self.culture.convictions[arguments_idx] += self.suggestibility * arguments_strength


class Environment(object):

    def __init__(self, n_agent, t_max, culture_length):

        # --- Parameters --- #
        self.t_max = t_max
        self.culture_length = culture_length
        self.n_agent = n_agent
        # ----------------- #

        self.agents = []

    def create_agents(self):

        for i in range(self.n_agent):
            a = Agent(
                proselytism=np.random.randint(self.culture_length),
                suggestibility=np.random.random(),
                convictions=np.random.random(self.culture_length) * 2 - 1
            )

            self.agents.append(a)

    def run(self):

        self.create_agents()

        for t in tqdm(range(self.t_max)):

            self.one_step()

    def one_step(self):

        # Take a random order among the indexes of the agents.
        random_order = np.random.permutation(self.n_agent)

        for i in random_order:

            # Each agent is "initiator' during one period.
            initiator = self.agents[i]

            # A 'responder' is randomly selected.
            responder = self.agents[np.random.choice(np.delete(np.arange(self.n_agent), i))]

            arg_idx, arg_strength = initiator.try_to_convince()
            responder.get_influenced(arguments_idx=arg_idx, arguments_strength=arg_strength)


class Experiment(object):

    def __init__(self, seed=None):

        self.arg_for_n_agent = [100]
        self.arg_for_t_max = [10]
        self.arg_for_culture_length = [4]
        self.seed = set_seed(seed)

    def run(self):

        print("Experiment: Running.")
        print()

        for culture_length, t_max, n_agent \
                in zip(self.arg_for_culture_length, self.arg_for_t_max, self.arg_for_n_agent):

            env = Environment(culture_length=culture_length, t_max=t_max, n_agent=n_agent)
            env.run()

    def save(self):

        pass

    def plot(self, results):

        pass


# ---------------- TEST FUNCTIONS ---------------- #


def test_culture():

    print("CULTURE DEMO")

    k = 5

    culture = Culture(convictions=np.random.random(k) * 2 - 1)

    print("First culture", culture)
    print("First convictions", culture.convictions)

    culture.convictions[:] = 0.3, - 0.4, 0.6, 0.8, -0.5

    print("New convictions", culture.convictions)
    print("New culture", culture)
    print("3 most robust convictions", culture.get_most_robust_convictions(n=3))


def test_influence():

    print("INFLUENCE DEMO")

    a = Agent(suggestibility=1, proselytism=3, convictions=[0.5, 0.3, 0.4, 0.2])

    print("Old culture of agent", a.culture)
    print("Old convictions of agent", a.culture.convictions)

    a.get_influenced(arguments_idx=[1, 3], arguments_strength=[-0.4, -0.4])

    print("New convictions of agent", a.culture.convictions)
    print("New culture of agent", a.culture)


def test_conviction():

    print("CONVICTION DEMO")

    a = Agent(suggestibility=1, proselytism=3, convictions=[0.5, 0.3, -0.4, 0.2])

    print("Culture of agent", a.culture)
    print("Convictions of agent", a.culture.convictions)

    best_convictions, values = a.try_to_convince()
    print("Best 3 convictions are:", best_convictions)
    print("Values for these convictions are:", values)


# ---------------- MAIN ---------------- #


def main():

    test_culture()
    print()
    print("*" * 10)
    print()
    test_influence()
    print()
    print("*" * 10)
    print()
    test_conviction()
    print()
    print("*" * 10)
    print()
    exp = Experiment(seed=None)
    exp.run()


if __name__ == "__main__":

    main()
