# Ensemble-Regime-Detector
This is a project that uses an Ensemble Model to detect market regime

The project uses an Hybrid Ensemble architecture that relies on statistical rigor and structural boundaries.
Here is the three-layer pipeline I’m currently testing:
1️)The Core: Hidden Markov Models (HMM)Instead of trying to guess infinite market states, an HMM focuses purely on the unobservable "latent" states of the market using robust, low-dimensional statistical properties (mean and variance). It models the transition probabilities between states, anchoring the strategy in actual market cycles rather than noise.
2️ The Quality Gate: CUSUM FilterStandard HMMs are notoriously sensitive to daily volatility, causing them to constantly flip-flop. To fix this, I feed features through a Cumulative Sum (CUSUM) filter. This acts as a barrier, ensuring the model only registers true structural breaks and ignores transient daily noise.
3️ The Execution Anchor: Hysteresis LayeringThe biggest hidden killer in regime-switching portfolios is transaction costs from whipsawing. By introducing hysteresis, I’ve added a "memory effect" and buffer zones. If the model needs a 65% probability to confirm a new regime, it requires it to drop below 35% to exit it. This prioritizes execution stability over reckless reactivity.
By combining the probabilistic power of HMMs with structural filters, the goal is to build a regime detector that is stable, interpretable, and mathematically insulated from overfitting.
