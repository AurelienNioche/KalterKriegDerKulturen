import matplotlib
matplotlib.use('TkAgg')
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
        Convictions inherits from numpy array.
        What it does in more is to modify the culture when it is modified itself.
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
        Culture inherits from numpy array.
        What it does in more is to have a 'Convictions' object as attribute
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

        # np.argsort() returns the indices that would sort the array
        # [::-1] reverse the sorting from increasing to decreasing order
        # [:n] selects the n first elements
        unique_convictions_values = np.unique(self.convictions)
        if len(unique_convictions_values) == len(self.convictions):
            return np.argsort(np.absolute(self.convictions))[::-1][:n]
        else:
            # Class in decreasing order the unique different values of convictions
            sorted_unique_values = sorted(unique_convictions_values, reverse=True)
            to_return = np.zeros(len(self.convictions), dtype=int)
            i = 0
            for value in sorted_unique_values:
                list_of_idx = np.where(self.convictions == value)[0]

                to_return[i: i + len(list_of_idx)] = np.random.permutation(list_of_idx)
                i += len(list_of_idx)

            return np.asarray(to_return)

        # temp_order = np.argsort(np.absolute(self.convictions))[::-1][:n]


class Agent(object):
    """ An agent is defined by its culture/convictions, suggestibility and proselytism.
    suggestibility and proselytism are supposedly 'innate' attributes,
    while culture and convictions are subject to changes while being influenced by other agents.

    convictions and culture are interdependent, culture is the 'binarized' (set {0, 1}) version
        of convictions vector (range [-1,1]).

    suggestibility: real in range [0, 1]
    proselytism: integer in range [0, culture size]
    convictions: vector of reals in range [-1, 1]
    culture: vector of boolean values
    """
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

    def apply_linear_ind_influence(self, arguments_idx, arguments_strength, verbose=False):
        """ Apply linear and independent influence formula. """
        if verbose:
            print("---apply_linear_ind_influence()")
            print("self.culture.convictions[arguments_idx]")
            print(self.culture.convictions[arguments_idx])
            print("self.suggestibility")
            print(self.suggestibility)
            print("arguments_strength")
            print(arguments_strength)
        self.culture.convictions[arguments_idx] += self.suggestibility * arguments_strength

    def apply_threshold_influence(self):
        # Apply threshold in order to not exceed the limits [-1,1]
        self.culture.convictions[self.culture.convictions<-1] = -1
        self.culture.convictions[self.culture.convictions>1] = 1

        # TODO: the following line with "np.clip()" does not work. It produces a strange error at a latter point in the program:
            # AttributeError: 'Convictions' object has no attribute 'parent'
        # self.culture.convictions = np.clip(self.culture.convictions, -1, 1)

    def get_influenced(self, arguments_idx, arguments_strength):
        """ Modify the own convictions of an agent based on the 'attacker's argument
        strength and the agent's own suggestibility.
        The formula is linear and is independent on the self convictions.

        Inputs:
            - arguments_idx: list of indices of arguments/?convictions # TO_CHECK
            - arguments_strength: list of same size as arguments_idx with the correspding strength of arguments of the attacker

        Ideas for further dev:
            suggestibility could be 2-fold depending if the attacker has arguments that goes in the same 'direction'."""
        # Apply influence formula
        self.apply_linear_ind_influence(arguments_idx, arguments_strength)
        # Apply threshold
        self.apply_threshold_influence()

    def get_influenced_nonlinear(self, arguments_idx, arguments_strength):
        """ Modify the own convictions of an agent based on the 'attacker's argument
        strength and the agent's own suggestibility.
        The formula is non-linear and is dependent on the self convictions.

        Now the influence is considered to be non-linear :
            - if self convictions are between -0.5 and +0.5,
                then the formula is the same than the "linear" version (i.e. get_influenced()).
            - if self convictions are below are above this range,
                if the attackers arguments are or same sign, the formulu
                then the self suggestibility gets weaker the more the convictions gets to the extremes (-1 or +1).
        suggestibility could be 2-fold depending if the attacker has arguments that goes in the same 'direction'."""
        # Apply influence formula
        for (i,s) in zip(arguments_idx, arguments_strength):
            if self.culture.convictions[i] >= -0.5 or self.culture.convictions[i] <= 0.5:
                self.apply_linear_ind_influence([i], [s])
            else:
                # If the self convitions are in the opposite direction than the attacker's
                if self.culture.convictions * s < 0:
                    # The influence is decreased the more the self convictions are closed to the extremes {-1 ; 1}
                    self.culture.convictions[i] += 2 * (1 - abs(self.culture.convictions[i])) \
                                                * self.suggestibility * s
                    # For abs(self.culture.convictions[i]) equal to 0.5 this is identical
                        # to the classical formula because the new factor equals 1: 2 * (1 - 0.5) = 1
                elif self.culture.convictions * s > 0:
                    # The influence is increased the more the self convictions are closed to the extremes {-1 ; 1}
                    if 1 - abs(self.culture.convictions[i]) < 10**-3:
                        self.culture.convictions[i] += 10**3 * self.suggestibility * s
                    else:
                        self.culture.convictions[i] += 1 / (2 * (1 - abs(self.culture.convictions[i]))) \
                                                * self.suggestibility * s
                else:
                    # Nothing happens in case it is equal to 0, because the product would be 0 anyway.
                    pass

        # Apply threshold
        self.apply_threshold_influence()


class Environment(object):

    def __init__(self, n_agent, t_max, culture_length, influence_type=None):

        # --- Parameters --- #
        self.t_max = t_max
        self.culture_length = culture_length
        self.n_agent = n_agent
        self.influence_type = influence_type
        if self.influence_type is None:
            self.influence_type = 'linear'
        # ----------------- #
        self.agents = []

        # Generation of random agents
        self.create_agents()

    def _assert_agent(self, agent):
        # sanity checks before influence round
        assert np.all(agent.culture.convictions >= -1)
        assert np.all(agent.culture.convictions <= 1)
        assert agent.suggestibility >= 0
        assert agent.suggestibility <= 1
        assert agent.proselytism >= 0
        assert agent.proselytism <= self.culture_length
        # TODO: add check if proselytism is an integer

    def get_matrix_of_agents_culture(self):
        return np.array([a.culture for a in self.agents])

    def get_matrix_of_agents_convictions(self):
        return np.array([a.culture.convictions for a in self.agents])

    def create_agents(self):

        # intialize list of agents
        self.agents = []

        # create random agents
        for i in range(self.n_agent):
            a = Agent(
                proselytism=np.random.randint(self.culture_length),
                suggestibility=np.random.random(),
                convictions=np.random.random(self.culture_length) * 2 - 1
            )

            self.agents.append(a)

    def create_orthogonal_agents(self):
        """ Create agents that have different cultures.
            Orthogonal means that one the culture space any agent should differ at least by one bit. """
        pass

    def run(self):

        for t in tqdm(range(self.t_max)):

            self.one_step()

    def one_step(self, verbose=False):

        # Take a random order among the indexes of the agents.
        random_order = np.random.permutation(self.n_agent)

        for i in random_order:

            # Each agent is "initiator' during one period.
            initiator = self.agents[i]
            # A 'responder' is randomly selected.
            responder = self.agents[np.random.choice(np.delete(np.arange(self.n_agent), i))]
            if verbose:
                print("Agent initiator: "+str(i))
                print("Agent influenced: "+str(self.agents.index(responder)))

            # Sanity checks/asserts
            self._assert_agent(initiator)
            self._assert_agent(responder)

            arg_idx, arg_strength = initiator.try_to_convince()

            if self.influence_type == 'linear':
                responder.get_influenced(arguments_idx=arg_idx, arguments_strength=arg_strength)
            elif self.influence_type == 'nonlinear':
                responder.get_influenced_nonlinear(arguments_idx=arg_idx, arguments_strength=arg_strength)
            else:
                str_err = "self.influence_type not set to a correct value: " + str(self.influence_type)
                raise Exception(str_err)

    def mult_cul_all_agents(self, factor):
        """ Multiply the culture of all agents by a given factor."""
        for c in [a.culture.convictions for a in self.agents]:
            c *= factor

    def make_agent_dictator(self, agent_indices):
        """ Modifiy all agents with the given indices to make them dictators.
            An agent become a dictator by getting a suggestibility of 0
                and a proselytism at the maximal value (i.e. the length of the culture)."""
        for i in agent_indices:
            # A dictator cannot change his believes/opinions
            self.agents[i].suggestibility = 0.
            # A dictator influences from all his culture
            self.agents[i].proselytism = self.culture_length

    def plot(self):
        self.plot_culture()
        self.plot_convictions()

    def plot_culture(self):
        # plt.figure()
        plt.matshow(self.get_matrix_of_agents_culture())
        plt.colorbar()
        plt.clim(0,1) # Sets the min/max limits of colorbar
        plt.title("Culture")

    def plot_convictions(self):
        # plt.figure()
        plt.matshow(self.get_matrix_of_agents_convictions())
        plt.colorbar()
        plt.clim(-1,1) # Sets the min/max limits of colorbar
        plt.title("Convictions")

class Experiment(object):

    def __init__(self, seed=None):

        self.n_agent_list = [100]
        self.t_max_list = [10]
        self.cul_len_list = [4] # culture_length
        self.seed = set_seed(seed)

    def run(self):

        print("Experiment: Running.")
        print()

        for culture_length, t_max, n_agent \
                in zip(self.cul_len_list, self.t_max_list, self.n_agent_list):

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

def main_dictatorship():
    def print_agents_cul():
        print("Cultures of agents:")
        print([a.culture for a in env.agents])
        print("Convictions of agents:")
        print([a.culture.convictions for a in env.agents])

    # set_seed(1)
    env = Environment(culture_length=30, t_max=500, n_agent=100, influence_type='linear')
    print("---init")
    print_agents_cul()

    # print("---lowered")
    # env.mult_cul_all_agents(factor=0.1)
    # print_agents_cul()

    env.plot()

    print("---make dictators")
    # env.make_agent_dictator([0])
    print_agents_cul()

    print("---run")
    env.run()

    print("---runned")
    print_agents_cul()
    print(env.get_matrix_of_agents_convictions())
    env.plot()
    plt.show()

if __name__ == "__main__":

    # main()
    main_dictatorship()
