# Kalten Krieg der Kulturen

## General definitions
- belief: vector of size k (same size as convictions) which is a combinaison of 3 values in the set {0, 1, #} for each possible convictions atoms / propositions

# Classes


## class kultur

### attributes
- convictions : vector of reals [-1, 1] of size k

### methods
- get() : return kultur which the binarization of convictions vector
  returned: vector of booleans {0, 1} of size k
- check_belief() : check if the agent has one particular belief, return a boolean


## class agent

### attributes
- proselytism [attack ability]: integer between 0 and k
- suggestibility [defense ability]:
- kultur (instance)

### methods
- try2convince : tries to change the convictions vector of another given agent
  - checks in it own convictions and its own proselytism and takes the n more robust propositions (i.e. the convictions atoms it is more convinced of, close to 1 or -1)
- being_influenced: modifies the convictions of the agents given the following rule:
> self.C <- self.C + self.S * other.C (applying a mask to S and C, given the kulturs)

## class environment
### attributes
- N : number of agents
- K : size of kultur/convictions
- list of agents

### methods
- matching: choice{random}
- create agents
- one_step: a step of the simulation, each agent tries to influence an other agent, and/ may be influenced by another one.

## class expriment
(create environments and launches simulations)

### methods
- run
- save
- plot_histogram_kulturen: plot the histogram of all kultures to see if some of them are much more dominant
> idea: organization of the x axis could be based on the resulting decimal interpretation of binary kultur vector

- plot_spectrogram_kulturen: similar as plot_histogram_kulturen but the x-axis is shifted to y-axis and y-axis (counts) is transformed into intensity color levels. New x-axis is the time.
