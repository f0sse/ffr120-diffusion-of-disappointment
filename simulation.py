#!/usr/bin/env python3

import numpy as np
import networkx as nx
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

from tqdm import tqdm
from enum import IntEnum
from joblib import Parallel, delayed
from argparse import ArgumentParser
from collections.abc import Sequence
from matplotlib.animation import FuncAnimation

################################################################################
##
## Parameters
##
################################################################################

# simulation
Δt = 0.1 # [day]
MAX_TIME = 50 # [day]
MAX_STEPS = int(MAX_TIME/Δt)
NUM_TRIALS = 20

# graph construction
NUM_AGENTS = 500
NUM_EDGES = 2
TRIANGE_PROBABILITY = 0.1

# marketing campaign
NUM_SEED = 5

# social interaction rate (events per day)
INTERACTION_RATE_MEAN = 10
INTERACTION_RATE_STD = 2

# probability that neighbor sentiment changes someones mind
# (sway is also affected by criticality, persuasiveness)
#SWAY_PROBABILITY = 0.1
SWAY_RATE = 2 # (events per day)
SWAY_PROBABILITY = -np.expm1(-SWAY_RATE*Δt)

# purchase rate (events per day)
PURCHASE_RATE_MEAN = 5
PURCHASE_RATE_STD = 0.2

# product breakdown
PRODUCT_LIFESPAN_MEAN = 10
PRODUCT_LIFESPAN_STD = 2

# when influencers stop advocating
CREDIBILITY_THRESHOLD = 0.3

# angular velocity of graph animation
ANGULAR_COEFFICIENT = 0.15

################################################################################
##
## Model
##
################################################################################

class State(IntEnum):
	Neutral=0
	Negative=1
	Positive=2
	Influence=3
	Silent=4
	TOTAL=5

def betaprob(size: tuple[int,int] = 1, a: float = 5, b: float = 5):
	"""
	Approximates a Gaussian distribution using the Beta distribution,
	to get probabilities in [0,1] with mean 0.5 using default a, b.
	"""
	return np.random.beta(a=a,b=b,size=size)


def rate2prob(r: float | np.ndarray) -> float | np.ndarray:
	"""
	Converts a continuous rate (events per unit time) into a probability
	for a discrete time step Δt.
	"""
	if (r < 0).any():
		raise ValueError("At least one rate is < 0")
	return -np.expm1(-r*Δt) # -(exp(...) - 1)


class Agents(np.ndarray):
	"""
	A hack-job of a subclass of ndarray, to make working with
	agents and their attributes much easier. There are many
	pseudo-attributes that exist to facilitate more readable code for
	selecting parts of the ndarray, such as "state". Do not think
	of those as attributes. They are data of the array. The
	only attributes in the classical sense are "chain" and "index"
	which are used to implement write-back mechanics when copies are
	created from advanced indexing.
	"""

	# Agent field indices
	STATE = 0
	SOCIAL = 1
	CRITICAL = 2
	PERSUASIVE = 3
	BUYRATE = 4
	BUYTIME = 5
	BREAKTIME = 6
	NFIELDS = 7

	# Pseudo-attributes for agent fields
	ATTRIBUTES = {
		"STATE": STATE,
		"SOCIAL": SOCIAL,
		"CRITICAL": CRITICAL,
		"PERSUASIVE": PERSUASIVE,
		"BUYRATE": BUYRATE,
		"BUYTIME": BUYTIME,
		"BREAKTIME": BREAKTIME,
	}

	def __new__(cls,
	            /,
	            obj=None,
	            shape=None,
	            *,
	            chain=None,
	            index=None,
	            copy=True,
	            ):
		"""
		To make sure chained copies also write back to the original instance,
		the parameters "chain" and "index" facilitate write-back mechanics.
		See the "__setitem__" method for details.
		"""
		if not all(a is None for a in [chain, index]):
			if not all(a is not None for a in [chain, index]):
				raise ValueError("Invalid use of proxying mechanics")
		if obj is None:
			if shape is None:
				raise ValueError("must specify 'shape'")
			# create a new instance
			crit1 = isinstance(shape, int | np.integer)
			crit2 = (
				isinstance(shape, Sequence)
				and
				all(isinstance(x, int | np.integer) and x>=0 for x in shape)
			)
			if not crit1 or crit2:
				err = "'shape' must be an integer or a sequence of integers"
				raise ValueError(err)
			if crit1 or len(shape) == 1:
				num = shape[0] if isinstance(shape, Sequence) else shape
				shape = (num, Agents.NFIELDS)
			obj = super().__new__(cls, shape=shape)
		else:
			# "copy" initialization
			if shape is not None:
				raise ValueError("both 'shape' and 'obj' is ambiguous")
			if not isinstance(obj, np.ndarray):
				raise ValueError("'obj' is not 'np.ndarray'")
			# shape of ndarray must be compatible
			crit0 = obj.ndim == 0 or  obj.size == 0
			crit1 = obj.ndim == 1 and obj.size == Agents.NFIELDS
			crit2 = obj.ndim == 2 and obj.shape[1] == Agents.NFIELDS
			if crit0 or crit1 or crit2:
				obj = np.asarray(obj, copy=copy, dtype=float).view(cls)
			else:
				print(obj.ndim, obj.size, type(obj))
				raise ValueError("'obj' has incompatible shape")
		# write-back attributes
		obj.chain = chain # <- parent from which self was created
		obj.index = index # <- indices used when -------||-------
		return obj

	def __array_finalize__(self, obj):
		if obj is None:
			# explicit constructor (super().__new__)
			return
		# instance of ndarray was cast into agents,
		# e.g. from slicing, view, etc.
		self.chain = getattr(obj, "chain", None)
		self.index = getattr(obj, "index", None)

	def __init__(self,
	             /,
	             obj=None,
	             shape=None,
	             *,
	             chain=None,
	             index=None,
	             copy=True,
	             ):
		# __new__ -> __array_finalize__ -> __init__,
		# with the same arguments as __new__.
		if obj is None:
			# self is a newly created Agents instance.
			# initialize agent population:
			N = self.as2d.shape[0]
			# states
			self.state = State.Neutral
			# rate
			self.social = rate2prob(np.random.normal(
				INTERACTION_RATE_MEAN,
				INTERACTION_RATE_STD,
				N,
			))
			self.buyrate = rate2prob(np.random.normal(
				PURCHASE_RATE_MEAN,
				PURCHASE_RATE_STD,
				N,
			))
			# probabilities
			self.critical = betaprob(N)
			self.persuasive = betaprob(N)
			# statistics
			self.breaktime = -1
			self.buytime = -1

	def __getitem__(self, key):
		sel = super().__getitem__(key)
		if not isinstance(sel, np.ndarray):
			return sel
		if np.may_share_memory(self, sel):
			return sel
		# sel is already a copy, so avoid double-copy
		return Agents(sel, copy=False, chain=self, index=key)

	def __setitem__(self, key, value):
		super().__setitem__(key, value)
		# assume that index is also set if chain is set
		if hasattr(self, "chain") and self.chain is not None:
			# propagate changes backwards
			self.chain[self.index] = self

	@property
	def as2d(self):
		return self[None,...] if len(self.shape) == 1 else self

	def __getattr__(self, attr: str):
		name = attr.upper()
		if name in self.ATTRIBUTES:
			return self.as2d[:, self.ATTRIBUTES[name]]
		err = f"'{self.__class__.__name__}' object has no attribute '{attr}'"
		raise AttributeError(err)

	def __setattr__(self, attr: str, value: object):
		name = attr.upper()
		if name in self.ATTRIBUTES:
			self.as2d[:, self.ATTRIBUTES[name]] = value
		else:
			super().__setattr__(attr, value)


class Simulation(object):
	def __init__(self,
	             /,
	             n=NUM_AGENTS,
	             m=NUM_EDGES,
	             p=TRIANGE_PROBABILITY,
	             *,
	             μ=PRODUCT_LIFESPAN_MEAN,
	             σ=PRODUCT_LIFESPAN_STD,
	             ):
		self.μ = μ
		self.σ = σ
		self.time = 0
		self.step = 0
		self.agents = Agents(shape=n)
		self.graph = nx.powerlaw_cluster_graph(n, m, p)
		self.layout = nx.spring_layout(self.graph, k=.2, dim=3)
		self.positions = np.asarray([self.layout[n] for n in self.graph.nodes])
		self._campaign_setup()

	def _influential_nodes(self, n=NUM_SEED):
		# determine influence using betweenness centrality
		centrality = nx.betweenness_centrality(self.graph)
		candidates = sorted(centrality, key=centrality.get, reverse=True)
		return candidates[:NUM_SEED]

	def _campaign_setup(self):
		candidates = self._influential_nodes()
		persuasion = betaprob(len(candidates), 6, 2)
		self.agents[candidates,Agents.STATE] = State.Influence
		self.agents[candidates,Agents.PERSUASIVE] = persuasion

	@property
	def finished(self):
		s = self.agents.state
		a = (s == State.Positive).any()
		b = (s == State.Influence).any()
		c = (s == State.Neutral).any()
		return not (a or b or c)

	@property
	def customers(self) -> np.generic:
		c = (self.agents.buytime >= 0) & (self.agents.breaktime > self.time)
		return np.sum(c)

	@property
	def purchases(self) -> np.generic:
		c = self.agents.buytime >= 0
		return np.sum(c)

	def countstate(self, state: State) -> np.generic:
		return np.sum(self.agents.state == state)

	def timestep(self):
		###########
		# 1. Sway
		###########

		swayed_positive = []
		swayed_negative = []

		swayable = [State.Neutral, State.Positive, State.Negative]

		for node in self.graph.nodes:
			agent = self.agents[node]

			if not (agent.state == swayable).any():
				# is influencer in either state
				continue
			if agent.buytime >= 0:
				# already bought the product
				continue

			conns = [*self.graph.neighbors(node)]
			neigh = self.agents[conns]

			# not really independent, but modeled as such for simplicity
			talksto = np.random.random() < neigh.social * agent.social

			if not talksto.any():
				continue

			neigh = neigh[talksto]

			influent = +1.*(neigh.state == State.Influence)
			positive = +1.*(neigh.state == State.Positive) + influent
			negative = -1.*(neigh.state == State.Negative)

			pos = np.average(positive, weights=neigh.persuasive)
			neg = np.average(negative, weights=neigh.persuasive)

			sentiment = pos + neg # -1 to +1
			intensity = abs(sentiment)

			Psway = SWAY_PROBABILITY * intensity * (1-agent.critical)

			if np.random.random() < Psway:
				if sentiment > 0:
					swayed_positive.append(node)
				if sentiment < 0:
					swayed_negative.append(node)

		self.agents[swayed_positive].state = State.Positive
		self.agents[swayed_negative].state = State.Negative

		###############
		# 2. Purchase
		###############

		mask = (self.agents.state==State.Positive)&(self.agents.buytime<0)
		spec = self.agents[mask]
		dice = np.random.random(spec.buyrate.shape)
		mask = dice < self.agents[mask].buyrate
		buys = spec[mask]
		buys.buytime = self.time
		buys.breaktime = np.random.normal(
			self.μ,
			self.σ,
			buys.breaktime.shape
		) + self.time

		################
		# 3. Breakdown
		################

		mask = (self.agents.state==State.Positive)
		posa = self.agents[mask]
		mask = (posa.buytime >= 0) & (posa.breaktime < self.time)
		posa[mask].state = State.Negative

		##################
		# 4. Credibility
		##################

		nodes = np.argwhere(self.agents.state == State.Influence).flat

		for node in nodes:
			conns = np.asarray([*self.graph.neighbors(node)])
			neigh = self.agents[conns]

			nmask = (neigh.state!=State.Influence)&(neigh.state!=State.Silent)
			neigh = neigh[nmask]
			conns = conns[nmask]

			# assume that influencers care about other people's
			# social status (in terms of "followers/connections")
			ranks = [len([*self.graph.neighbors(i)]) for i in conns]

			negative = 1.*(neigh.state == State.Negative)
			sentiment = np.average(negative, weights=neigh.persuasive*ranks)

			if sentiment > 1-CREDIBILITY_THRESHOLD:
				self.agents[node].state = State.Silent

		# increment
		self.time += Δt
		self.step += 1

################################################################################
##
## Simulate / Animate
##
################################################################################

class Animation(FuncAnimation):
	COLOR_MAP = {
		State.Neutral:   'grey',
		State.Negative:  'tab:red',
		State.Positive:  'tab:green',
		State.Influence: 'gold',
		State.Silent:    'tab:blue'
	}
	def __init__(self):
		self.fig = plt.figure(figsize=(7, 7))
		self.ax = self.fig.add_subplot(111, projection='3d')
		self.fig.subplots_adjust(left=-0.2, right=1.2, bottom=-0.25, top=1.2)
		self.ax.set_axis_off()
		self.ax.set(aspect="equal")
		self.ax.view_init(elev=20, azim=0)
		self.sim = Simulation()
		# nodes
		self.xyz = (
			self.sim.positions[:,0],
			self.sim.positions[:,1],
			self.sim.positions[:,2],
		)
		self.nodeplot = self.ax.scatter(
			*self.xyz,
			s=self._get_sizes(),
			c=self._get_colors(),
			edgecolors='k',
			lw=0.2,
			alpha=.85,
		)
		# edges
		colors = self._get_colors()
		self.edges = self._get_edges()
		self.edgeplot, = self.ax.plot(*self.edges, c='k', alpha=.3, lw=.35)
		super().__init__(
			self.fig,
			self.update,
			init_func=self.initialize,
			cache_frame_data=False,
			frames=self.stopper,
			interval=1,
			repeat=False,
			blit=False,
		)

	##
	## Helpers
	##

	def _get_edges(self):
		ex, ey, ez = [], [], []
		for u, v in self.sim.graph.edges:
			x1,y1,z1 = self.sim.positions[u]
			x2,y2,z2 = self.sim.positions[v]
			ex.extend([x1,x2,np.nan])
			ey.extend([y1,y2,np.nan])
			ez.extend([z1,z2,np.nan])
		return ex,ey,ez

	def _get_colors(self):
		state = self.sim.agents.state
		return list(map(lambda x: Animation.COLOR_MAP[x], state))

	def _get_sizes(self):
		state = self.sim.agents.state
		persuasive = self.sim.agents.persuasive
		sizes = np.where(state == State.Influence, 300, 100)
		return persuasive * sizes

	##
	## Properties
	##

	def get_figure(self):
		return self.fig

	@property
	def artists(self):
		return [self.nodeplot,self.edgeplot]

	##
	## Animation
	##

	def stopper(self):
		i = 0
		while self.sim.time < MAX_TIME and not self.sim.finished:
			print(f"\rtime = {self.sim.time:.4f}", end='')
			yield i
			i += 1
		print(" Done")

	def initialize(self):
		#self.fig.savefig(
		#	f"img/snapshot_t{self.sim.time:.0f}.pdf",
		#	bbox_inches='tight',
		#	pad_inches=-1.5,
		#	transparent=False)
		return self.artists

	def update(self, frame):
		self.sim.timestep()
		self.ax.view_init(elev=20, azim=ANGULAR_COEFFICIENT*frame)
		self.nodeplot.set_facecolor(self._get_colors())
		self.nodeplot.set_sizes(np.asarray(self._get_sizes()))
		#if self.sim.step % int(5/Δt) == 0:
		#	self.fig.savefig(
		#		f"img/snapshot_t{self.sim.time:.0f}.pdf",
		#		bbox_inches='tight',
		#		pad_inches=-1.5,
		#		transparent=False)
		return self.artists

################################################################################
##
## Produce Graphs for Results
##
################################################################################

class Grapher(object):
	def __init__(self, /, μ=PRODUCT_LIFESPAN_MEAN, σ=PRODUCT_LIFESPAN_STD):
		super().__init__()
		self.μ = μ
		self.σ = σ
		# points of interest
		self.tippoint = np.ones(NUM_TRIALS) * np.nan
		self.colpoint = np.ones(NUM_TRIALS) * np.nan
		# histories (MAX_STEPS+1 to cover initial state)
		self.customers = np.ones((NUM_TRIALS,MAX_STEPS+1)) * np.nan
		self.purchases = np.ones((NUM_TRIALS,MAX_STEPS+1)) * np.nan
		self.positive = np.ones((NUM_TRIALS,MAX_STEPS+1)) * np.nan
		self.negative = np.ones((NUM_TRIALS,MAX_STEPS+1)) * np.nan
		self.neutral = np.ones((NUM_TRIALS,MAX_STEPS+1)) * np.nan

	def simulate(self, i: int, Q: mp.Queue):
		customers = []
		purchases = []
		positive = []
		negative = []
		neutral = []
		sim = Simulation(μ=self.μ, σ=self.σ)
		for _ in range(MAX_STEPS+1):
			customers.append(sim.customers)
			purchases.append(sim.purchases)
			positive.append(sim.countstate(State.Positive))
			negative.append(sim.countstate(State.Negative))
			neutral.append(sim.countstate(State.Neutral))
			if sim.finished:
				break
			sim.timestep()
			Q.put((i, 1))
		Q.put((i,None))
		pos = np.asarray(positive)
		neg = np.asarray(negative)
		pur = np.asarray(list(reversed(purchases)))
		getitem = lambda x: x.item() if x.size == 1 else -1
		tippingpoint = getitem(np.argwhere(pos<neg)[:1]) * Δt
		collapsepoint = getitem(pur.size-np.argwhere(np.diff(pur)!=0)[:1]) * Δt
		return (
			tippingpoint,
			collapsepoint,
			np.vstack((
				customers,
				purchases,
				positive,
				negative,
				neutral,
			))
		)

	def gather(self):
		"""
		Gather data from many simulations.
		"""
		T = (MAX_STEPS+1) * NUM_TRIALS
		Q = mp.Manager().Queue()
		S = np.zeros(NUM_TRIALS, dtype=np.uintp)
		# return as generator -> does not block
		G = Parallel(n_jobs=-1, return_as="generator")(
			delayed(self.simulate)(i, Q) for i in range(NUM_TRIALS)
		)
		with tqdm(total=T, leave=False) as pbar:
			while S.sum() < T:
				i, step = Q.get()
				if step is None:
					# a simulation finished
					S[i] = MAX_STEPS+1
					pbar.n = S.sum()
					pbar.refresh()
				else:
					S[i] += 1
					pbar.update(1)
		# G is a generator of NUM_TRIALS self.simulate(...)
		for i, result in enumerate(G):
			tippingpoint, collapsepoint, histories = result
			# might have ended early, slice until M
			M = histories.shape[1]
			# points of interest
			setitem = lambda x: np.nan if x < 0 else x
			self.tippoint[i] = setitem(tippingpoint)
			self.colpoint[i] = setitem(collapsepoint)
			# histories
			self.customers[i,:M] = histories[0,:M]
			self.purchases[i,:M] = histories[1,:M]
			self.positive[i,:M] = histories[2,:M]
			self.negative[i,:M] = histories[3,:M]
			self.neutral[i,:M] = histories[4,:M]

	def plot(self) -> tuple:
		"""
		Produce the graphs used for results.
		"""
		c_neg = Animation.COLOR_MAP[State.Negative]
		c_pos = Animation.COLOR_MAP[State.Positive]
		c_neu = Animation.COLOR_MAP[State.Neutral]
		c_cst = 'tab:purple'
		c_pur = 'tab:olive'
		style = {
			"lw": 1.5,
			"alpha": 0.8,
			"ls": '-',
		}

		time = np.arange(MAX_STEPS+1) * Δt

		customers = np.nanmean(self.customers, axis=0)
		purchases = np.nanmean(self.purchases, axis=0)
		positive = np.nanmean(self.positive, axis=0)
		negative = np.nanmean(self.negative, axis=0)
		neutral = np.nanmean(self.neutral, axis=0)

		tippoint = np.nanmean(self.tippoint)
		colpoint = np.nanmean(self.colpoint)

		tipy = negative[np.where(time > tippoint)][:1].item()
		coly = purchases[np.where(time > colpoint)][:1].item()

		plt.rc("font", size=12)
		plt.rc("axes", titlesize=16, labelsize=14)
		plt.rc("xtick", labelsize=12)
		plt.rc("ytick", labelsize=12)
		plt.rc("legend", fontsize=12)

		fig, ax = plt.subplots(figsize=(8,4.5),layout="constrained")

		ax.plot(time, customers, c=c_cst, **style, label="customers")
		ax.plot(time, purchases, c=c_pur, **style, label="purchases")
		ax.plot(time, positive, c=c_pos, **style, label="positive")
		ax.plot(time, negative, c=c_neg, **style, label="negative")
		ax.plot(time, neutral, c=c_neu, **style, label="neutral")
		ax.plot([tippoint],[tipy],'*',ms=8,c=c_neg)
		ax.plot([colpoint],[coly],'*',ms=8,c=c_pur)
		ax.axvline(self.μ, ls='--', lw=0.9, c='k', alpha=.8)
		ax.set(xlabel="$t$", ylabel="$N$")
		ax.set(title=f"$\\mu = {self.μ}$, $\\sigma = {self.σ}$")
		ax.grid(True, alpha=.3)
		ax.legend()

		return fig, ax

################################################################################
##
## Program Flow
##
################################################################################

def main_gui():
	a = Animation()
	plt.show()

def main_nogui():
	M = np.linspace(1,20,20)
	Σ = np.linspace(1,20,20)/5
	tp = []
	cp = []
	for u, s in zip(M, Σ):
		g = Grapher(u, s)
		g.gather()
		tp.append(np.nanmean(g.tippoint))
		cp.append(np.nanmean(g.colpoint))
		fig, ax = g.plot()
		#fig.savefig(
		#	f"img/graph_u{u:.0f}_s{s:.1f}.pdf",
		#	bbox_inches='tight',
		#	pad_inches=.4,
		#	transparent=False)
		plt.show()
	# tipping point
	fig, ax = plt.subplots(figsize=(8,4.5), layout="constrained")
	ax.plot(M, tp, '--', marker='o', lw=2, ms=8)
	ax.set(xlabel="$\\mu$")
	ax.set(ylabel="$\\left\\langle T_\\text{tip}\\right\\rangle$")
	ax.set(title=f"Tipping point vs. mean breakdown time")
	ax.grid(True, alpha=.3)
	plt.show()
	#fig.savefig(
	#	f"img/tip_vs_breakdown.pdf",
	#	bbox_inches='tight',
	#	pad_inches=.4,
	#	transparent=False)
	# collapse point
	fig, ax = plt.subplots(figsize=(8,4.5), layout="constrained")
	ax.plot(M, cp, '--', marker='o', lw=2, ms=8)
	ax.set(xlabel="$\\mu$")
	ax.set(ylabel="$\\left\\langle T_\\text{col}\\right\\rangle$")
	ax.set(title=f"Market collapse vs. mean breakdown time")
	ax.grid(True, alpha=.3)
	plt.show()
	#fig.savefig(
	#	f"img/col_vs_breakdown.pdf",
	#	bbox_inches='tight',
	#	pad_inches=.4,
	#	transparent=False)
	# combined
	fig, ax = plt.subplots(figsize=(8,4.5), layout="constrained")
	ax.plot(M, tp, '--', marker='o', lw=2, ms=8, label="tipping point")
	ax.plot(M, cp, '--', marker='o', lw=2, ms=8, label="market collapse")
	ax.set(xlabel="$\\mu$")
	ax.set(ylabel="$\\left\\langle T\\right\\rangle$")
	ax.set(title=f"Event time vs. mean breakdown time")
	ax.grid(True, alpha=.3)
	ax.legend()
	plt.show()
	#fig.savefig(
	#	f"img/tip_col_vs_breakdown.pdf",
	#	bbox_inches='tight',
	#	pad_inches=.4,
	#	transparent=False)


if __name__ == "__main__":
	#
	# provide two modes:
	#   - FuncAnimation to show evolution
	#   - No GUI, to produce graphs for results
	#
	argp = ArgumentParser(
		prog="Simulation",
		description="Agent-based model of a social media marketing campaign.",
		epilog="The default is to produce graphs showing results."
	)
	argp.add_argument('-g', '--gui', action='store_true', help="show GUI")
	args=argp.parse_args()
	try:
		main = main_gui if args.gui else main_nogui
		main()
	except KeyboardInterrupt:
		pass
