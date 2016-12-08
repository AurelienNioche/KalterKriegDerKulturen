# Kalten Krieg der Kulturen

belief: vector of size k (same size as convictions) which is a combinaison of 3 values in the set {0, 1, #} for each possible convictions atoms / propositions

# classes
## class kultur
attr: convictions : vector of reals [-1, 1] of size k
meth: get() : return kultur which the binarization of convictions vector
  returned: vector of booleans {0, 1} of size k
meth: check_belief() : check if the agent has one particular belief, return a boolean

## class agent
attr: proselytism [attack ability]: integer between 0 and k
attr: suggestibility [defense ability]:
attr: kultur (instance)
meth: try2convince : tries to change the convictions vector of another given agent
  - checks in it own convictions and its own proselytism and takes the n more robust propositions (i.e. the convictions atoms it is more convinced of, close to 1 or -1)
meth: being_influenced: modifies the convictions of the agents givent the following rule:
  self.C <- self.C + self.S * other.C (applying a mask to S and C, given the kulturs)

## class environment
attr: N : number of agents
attr: K : size of kultur/convictions
attr: list of agents
meth: matching: choice{random}
meth: create agents
meth: one_step: a step of the simulation, each agent tries to influence an other agent, and/ may be influenced by another one.

## class expriment
(create environments and launches simulations)
meth: run
meth: save
meth: plot_histogram_kulturen: plot the histogram of all kultures to see if some of them are much more dominant
  idea: organization of the x axis could be based on the resulting decimal interpretation of binary kultur vector
meth: plot_spectrogram_kulturen: similar as plot_histogram_kulturen but the x-axis is shifted to y-axis and y-axis (counts) is transformed into intensity color levels. New x-axis is the time.
