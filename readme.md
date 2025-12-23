# Diffusion of Disappointment

An agent-based simulation model that explores how negative sentiment spreads through a social network following a marketing campaign for a faulty product. This simulation demonstrates the dynamics of opinion formation, product adoption, and reputation collapse in social networks.

## Overview

This simulation models a social media marketing campaign where influential agents initially promote a product. As users purchase and experience product breakdowns, negative sentiment diffuses through the network, potentially leading to a "tipping point" where negative opinions overtake positive ones, and ultimately a market collapse.

## Features

- **Agent-based modeling**: Simulates individual agents with unique characteristics (social activity, criticality, persuasiveness)
- **Network topology**: Uses a power-law cluster graph to represent realistic social network structures
- **Opinion dynamics**: Models how agents influence each other's opinions through social interactions
- **Product lifecycle**: Includes purchase behavior and product breakdown mechanics
- **Influencer credibility**: Influencers stop promoting when negative sentiment reaches a threshold
- **Visualization**: 3D animated visualization of the network and sentiment spread
- **Statistical analysis**: Batch simulations to analyze tipping points and market collapse timing

## Installation

### Requirements

Python 3.x with the following packages:

```bash
pip install -r requirements.txt
```

Dependencies include:
- numpy: Numerical computations
- networkx: Graph generation and analysis
- matplotlib: Visualization and animation
- pandas: Data manipulation
- joblib: Parallel processing
- tqdm: Progress bars

## Usage

The simulation offers two modes: GUI visualization and batch analysis.

### GUI Mode (Animation)

Watch the sentiment spread through the network in real-time:

```bash
python simulation.py --gui
```

This displays a 3D animated visualization where:
- **Gold nodes**: Influencers promoting the product
- **Green nodes**: Agents with positive sentiment
- **Red nodes**: Agents with negative sentiment (after product breakdown)
- **Grey nodes**: Neutral agents
- **Blue nodes**: Silent influencers (lost credibility)
- **Node size**: Indicates persuasiveness (larger = more persuasive)

### Batch Analysis Mode (Default)

Generate statistical graphs and analyze trends:

```bash
python simulation.py
```

This runs multiple simulations with varying parameters and produces graphs showing:
- Number of customers, purchases, and sentiment distribution over time
- Tipping point: When negative sentiment overtakes positive sentiment
- Market collapse: When purchases stop growing
- Relationship between product lifespan and critical events

## Model Components

### Agent States

Agents can be in one of five states:

1. **Neutral** (grey): Has not formed an opinion about the product
2. **Positive** (green): Has positive sentiment, may purchase the product
3. **Negative** (red): Has negative sentiment (typically after product breakdown)
4. **Influence** (gold): Initial influencers promoting the product
5. **Silent** (blue): Influencers who stopped promoting due to negative feedback

### Agent Attributes

Each agent has the following characteristics:

- **Social**: Probability of interacting with neighbors per time step
- **Critical**: Resistance to being swayed by others (0-1)
- **Persuasive**: Ability to influence others' opinions (0-1)
- **Buy rate**: Probability of purchasing when having positive sentiment
- **Buy time**: When the agent purchased (if applicable)
- **Break time**: When the product breaks down (sampled from normal distribution)

### Simulation Mechanics

The simulation proceeds in discrete time steps, with four key phases:

1. **Sway**: Agents interact with neighbors and may change opinions based on:
   - Sentiment of neighbors (positive/negative weights)
   - Persuasiveness of neighbors
   - Agent's own criticality (resistance to influence)
   - Interaction probabilities

2. **Purchase**: Agents with positive sentiment may purchase the product based on their buy rate

3. **Breakdown**: Products break down after a random lifespan (normal distribution), causing owners to develop negative sentiment

4. **Credibility**: Influencers monitor their neighbors' sentiment and stop promoting if too much negative feedback accumulates

### Key Parameters

Configurable parameters at the top of `simulation.py`:

**Simulation timing:**
- `Δt`: Time step size (0.1 days)
- `MAX_TIME`: Maximum simulation duration (50 days)
- `NUM_TRIALS`: Number of simulations for batch analysis (20)

**Network structure:**
- `NUM_AGENTS`: Number of agents in the network (500)
- `NUM_EDGES`: Edges per node in power-law graph (2)
- `TRIANGE_PROBABILITY`: Clustering coefficient (0.1)

**Marketing:**
- `NUM_SEED`: Number of initial influencers (5)

**Behavior rates:**
- `INTERACTION_RATE_MEAN/STD`: Social interaction frequency
- `SWAY_RATE`: Rate at which agents may change opinions
- `PURCHASE_RATE_MEAN/STD`: Purchase probability

**Product quality:**
- `PRODUCT_LIFESPAN_MEAN`: Average time until breakdown (10 days)
- `PRODUCT_LIFESPAN_STD`: Standard deviation (2 days)

**Influencer behavior:**
- `CREDIBILITY_THRESHOLD`: Negative sentiment threshold for influencers to stop (0.3)

## Model Interpretation

The simulation reveals several key phenomena:

- **Tipping Point**: The time when negative sentiment overtakes positive sentiment, typically occurring shortly after the mean product breakdown time
- **Market Collapse**: When purchase growth stops, indicating the campaign has failed
- **Influencer Cascade**: As products break down, influencers face increasing negative feedback and may go silent, accelerating the collapse
- **Network Effects**: The power-law structure means some agents are more connected and influential than others

## Associated Report

This code is part of the research report "Diffusion of Disappointment," which analyzes how negative experiences with products can spread through social networks and undermine marketing campaigns.

## Technical Notes

- The `Agents` class is a custom numpy array subclass that facilitates easier manipulation of agent attributes
- Multiprocessing is used for batch simulations to speed up parameter sweeps
- The network layout uses a 3D spring layout for visualization
- Time is modeled in continuous days with discrete time steps

