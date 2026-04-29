## Deep Learning for Financial Time Series: A Large-Scale

## Benchmark of Risk-Adjusted Performance

#### ∗

#### Adir Saly-Kaufmann^1 , Kieran Wood^1 ,^2 , Jan Peter-Calliess^1 , Stefan Zohren^1

(^1) Machine Learning Research Group, Department of Engineering Science, University of Oxford
(^2) Oxford-Man Institute of Quantitative Finance, University of Oxford

#### adir.saly-kaufmann@eng.ox.ac.uk, kieran.wood@eng.ox.ac.uk

#### janpeter.calliess@eng.ox.ac.uk, stefan.zohren@eng.ox.ac.uk

## Abstract

```
We present a large-scale benchmark of modern deep
learning architectures for a financial time-series pre-
diction and position sizing task, with a primary fo-
cus on Sharpe-ratio optimization. Evaluating linear
models, recurrent networks, transformer-based archi-
tectures, state-space models, and recent sequence-
representation approaches, we assess out-of-sample
performance on a daily futures dataset spanning com-
modities, equity indices, bonds, and FX spanning
2010–2025. Our evaluation goes beyond average re-
turns and includes statistical significance, downside
and tail-risk measures, breakeven transaction-cost
analysis, robustness to random seed selection, and
computational efficiency. We find that models ex-
plicitly designed to learn rich temporal representa-
tions consistently outperform linear benchmarks and
generic deep learning models, which often lead the
ranking in standard time-series benchmarks. Hybrid
models such asVSN+LSTM, a combination of Vari-
able Selection Networks (VSN) and LSTMs, achieves
the highest overall Sharpe ratio, whileVSN+xLSTM
and LSTM+PatchTST exhibit superior downside-
adjusted characteristics. xLSTM demonstrates the
largest breakeven transaction cost buffer, indicating
improved robustness to trading frictions.
```

```
∗Code is available upon request.
```

## 1 Introduction

```
In recent years, many new deep learning architectures
have emerged in the context of time-series forecast-
ing, with Transformer-based architectures [1] draw-
ing particular attention [2]. Several theoretical ex-
tensions and adaptations have been proposed, such
as iTransformer [3], PatchTST [4], and the Temporal
Fusion Transformer [5]. Each of these models aims
to address different challenges encountered in real-
world forecasting. For example, PatchTST improves
robustness by bundling data points and interpreting
features independently, whereas iTransformer focuses
on learning relationships between features without re-
lying on temporal order.
State Space Models have also presented an alter-
native to transformer-based architectures. In par-
ticular, Mamba2 [6] claims to achieve a mathemat-
ically principled architecture with linear-attention-
like behavior [7] while supporting arbitrarily large
lookback windows. More broadly, State Space Mod-
els have evolved from the theory of HiPPO (High-
order Polynomial Projection Operators) matrices [8],
which maintain compressed representations of past
information as the model evolves.
Historically, recurrent neural networks (RNNs) [9]
have been the most widely used deep-learning ar-
chitectures for time-series forecasting [10], especially
```

# arXiv:2603.01820v1 [q-fin.TR] 2 Mar 2026

Figure 1: End-to-end portfolio optimization pipeline:
Statistical and technical indicators are extracted from his-
torical close prices, serving as the predictive model’s in-
puts. The model outputs are transformed into portfolio
weights via a linear projection followed by a hyperbolic
tangent activation. Training is performed by minimizing
the negative Sharpe Ratio.

LSTMs [11]. More recently, the xLSTM architec-
ture [12] has been introduced as a new state-of-the-
art model, with additional gains demonstrated by
PsLSTM [13]. xLSTM replaces LSTM’s traditional
sigmoid gating with exponential gating and a normal-
ization term, while also employing a memory matrix
rather than a scalar value. PsLSTM further inte-
grates patching (similar to PatchTST) into the xL-
STM architecture.

Several benchmark studies have explored deep-
learning models for time-series forecasting [14], focus-
ing on applications such as weather prediction, elec-
tricity transformer temperatures, and transportation
data. These studies indicate that simple architec-
tures such as DLinear [15] can perform comparably to
more complex transformer-based models. Similarly,
[15] has shown that NLinear performs slightly bet-
ter in some settings. However, these datasets exhibit

```
strong seasonality and a high signal-to-noise ratio, in
contrast to financial time series [16].
Forecasting financial time series essentially reduces
to performing auto-regressive modeling [17], typically
formulated as
```

```
xt+1= f (xt,xt− 1 ,xt− 2 ,...) (1)
```

```
for each 0 ≤ t < T, where xtis the value of the pro-
cess at time t, T is the last time-step to predict, and
f (·) denotes a model-dependent forecasting function
that maps past observations to a one-step-ahead pre-
diction. For example, in AR(p) it is of the form:
```

```
f (xt,...,xt−p) =
```

```
Xp
```

```
i=
```

```
φixt−i,
```

```
where {φi} are fixed coefficients [17] and in
LSTM [11], f is instead represented by a nonlinear
recurrent mapping with learnable gating mechanisms
and hidden states B.4.
Financial return series are characterized by strong
noise, weak and time-varying predictability, and pro-
nounced non-stationarity. These properties imply
that predictive success hinges less on raw model ca-
pacity than on the ability to recover economically
meaningful signals from highly volatile data. In this
context, a successful model must simultaneously (i)
improve the signal-to-noise ratio by filtering out tran-
sient fluctuations, (ii) learn asset-specific dynamics
rather than imposing homogeneous temporal struc-
tures, and (iii) embed temporal dependencies in a
manner that remains stable across market regimes
[18, 19]. Several of the benchmarked architectures
incorporate explicit design choices aimed at enhanc-
ing one or more of these properties, including fea-
ture selection, structured state representations, and
temporal aggregation mechanisms [20]. We evaluate
both the incremental value of such architectural en-
hancements and their performance relative to models
that natively embed these characteristics, with par-
ticular emphasis on robustness across time and eco-
nomic regimes.
```

```
Contributions. This paper presented a unified
benchmark of modern deep learning architectures
```

for financial time-series prediction under a Sharpe-
ratio optimization framework. Using 15 years of data
spanning multiple asset classes and market regimes,
we evaluated linear models, recurrent networks,
transformer-based architectures, state-space models,
and recent sequence-representation approaches across
return, risk, robustness, and computational dimen-
sions.

Linear dynamics alone appear insufficient.
While linear models occasionally performed com-
petitively in specific high-volatility subperiods, they
failed to deliver stable performance across time and
provided limited incremental value relative to a pas-
sive benchmark. This supports the view that finan-
cial returns exhibit structural features not fully cap-
tured by linear autoregression.

Architectural inductive bias is decisive. Non-
linear models improved average performance, but
outcomes varied substantially across architectures.
Generic transformers and state-space models dis-
played heterogeneous, regime-sensitive behavior. In
contrast, VLSTM—designed to learn structured tem-
poral representations—delivered consistently strong
and stable risk-adjusted returns, suggesting that rep-
resentation compression, adaptive memory, and tem-
poral abstraction are particularly valuable in low
signal-to-noise environments.

Robustness and risk control matter as much
as returns. Downside exposure, tail behavior, and
stability under reduced seed aggregation materially
affected model suitability. VLSTM-based strategies
combined competitive returns with moderate draw-
downs and remained stable under weaker experimen-
tal budgets, indicating that performance was not
driven solely by favorable seed selection.

Asymptotic efficiency does not guarantee em-
pirical superiority. Although state-space models
offer attractive theoretical complexity, empirical ef-
fectiveness depended more strongly on inductive bias
than on asymptotic scaling alone.

Overall, the results suggest that effective financial
forecasting models benefit from jointly denoising re-
turns, learning asset-specific and regime-aware dy-
namics, and encoding temporal structure in a stable

```
and adaptive manner. While the conclusions remain
conditional on the dataset and backtesting protocol
employed, this benchmark provides a transparent ref-
erence point for future research in deep learning for
finance.
```

## 2 Architectures

#### 2.1 Problem Setup

```
Let {xt}Tt=1, xt∈ Rd, denote a multivariate time se-
ries of end of day features derived from commodity
futures, foreign-exchange (FX) futures, bonds, index,
and energy products, including prices, returns, and
technical indicators.
Given a fixed lookback window of length L, the
objective is to learn a function
```

```
fθ: RL×d→ [− 1 , 1], (2)
```

```
mapping historical observations
```

```
Xt= [xt−L+1,...,xt] (3)
```

```
to a scalar forecast or trading signal used to construct
daily positions,where 1 is the upper bound for a full
long position and -1 is the lower bound for a full short
position.
```

```
Trading Signal Generation To systematically
benchmark different deep learning paradigms, we
structure the function fθas a modular, two-stage
pipeline. The first stage consists of a candidate se-
quence architecture gφ(e.g., LSTM, PatchTST, or
Mamba2), which processes the input window to ex-
tract a temporal state representation of fixed hidden
dimension H:
```

```
ht= gφ(Xt), ht∈ RH. (4)
```

```
The second stage is a unified projection head ap-
plied to the terminal hidden state ht. This consists of
a linear transformation followed by a hyperbolic tan-
gent (tanh) activation function to bound the output:
```

```
ˆty = tanh
```

######

```
w⊤linht+ blin
```

######

###### , (5)

where wlin∈ RHand blin∈ R are learnable weights.
All models are trained using rolling windows and
evaluated in a fully out-of-sample trading framework.
We added ticker embeddings to all the mod-
els to enhance the learning per individual ticker/as-
set [21].

Portfolio Construction The scalar output ˆt,ky ∈
[− 1 , 1] generated by the projection head represents
the model’s directional conviction for asset k at time
t. Because financial assets exhibit vastly different
baseline volatilities, we employ a volatility target-
ing framework to equalize risk contributions across
the universe [22–24]. We estimate the ex-ante con-
ditional volatility σt,kfor each asset using an Ex-
ponentially Weighted Moving Average (EWMA) es-
timator (detailed in Appendix A). This estimation
induces a time-varying leverage factor, defined as
vs_factort,k = σ^1 t,k, which dynamically scales po-

sition sizes in response to shifting market regimes.
Given a constant target portfolio volatility σtgt(set
to 10% in our empirical evaluation), the realized port-
folio weight wt,kallocated to asset k is obtained by
scaling the neural network’s signal by this leverage
factor:

```
wt,k= ˆt,ky
```

######

```
σtgt
σt,k
```

######

```
= ˆt,ky · σtgt· vs_factort,k (6)
```

Given these target weights, the daily gross strategy
return for a specific asset k realized at time t + 1 is
the product of the position taken at the end of day t
and the subsequent asset return rt+1,k:

```
Rt+1,k= wt,k· rt+1,k (7)
```

Assuming an equal risk capital allocation across the
K active assets, the aggregate daily gross portfolio
return Rtport+1 is the cross-sectional average of the in-
dividual strategy returns:

```
Rportt+1=
```

###### 1

###### K

###### XK

```
k=
```

```
Rt+1,k (8)
```

End-to-end Optimization Unlike traditional
forecasting models that minimize predictive errors,

```
our framework directly optimizes for risk-adjusted
economic performance [24]. To train the parameters
θ of the network fθ, we compute the Sharpe Ratio
over a given training sequence of length T. Let
Rport = {Rport 1 ,...,RportT } represent the sequence
of daily portfolio returns. We define the sample
estimators for the expected returnˆE[R] and variance
Varˆ[R] of the portfolio as:
```

###### Eˆ[R] =^1

###### T

###### XT

```
t=
```

```
Rportt , (9a)
```

```
Var[ˆ R] =^1
T
```

###### XT

```
t=
```

######

```
Rportt −Eˆ[R]
```

###### 2

. (9b)

```
The loss function L(θ) is defined as the negative dif-
ferentiable annualized Sharpe Ratio:
```

```
L(θ) =−SR[cR] =−
```

###### ˆE[R]

```
q
Var[ˆ R] + ε
```

###### √

###### 252. (10)

```
where 252 represents the approximate number of
trading days in a year, and ε is a small constant added
for numerical stability. By minimizing this loss, the
network explicitly learns representations that maxi-
mize expected returns while heavily penalizing vari-
ance. Following the regime-robust DeePM framework
[25], we compute the Sharpe ratio objective onpooled
portfolio returns concatenating all sequences in the
batch, following their argument that this is the best
proxy for optimising out-of-sample Sharpe ratio. See
the end-to-end optimization pipeline in Figure 1.
```

```
Net Returns and Breakeven Transaction Costs
To account for implementation frictions, the net port-
folio return Rnett+1is defined by deducting the costs
associated with portfolio turnover:
```

```
Rnett+1= Rportt+1−
```

###### 1

###### K

###### XK

```
k=
```

```
ck|wt,k− wt− 1 ,k| (11)
```

```
where ckrepresents the proportional transaction cost
per unit of traded weight for asset k. Because real-
istic execution costs vary drastically across the as-
set universe (e.g., highly liquid short-term interest
```

rates versus illiquid agricultural commodities), im-
posing static ex-ante assumptions for ckcan severely
distort cross-sectional performance metrics. There-
fore, for model optimization and primary evaluation,
we set ck= 0 for all k to assess the pure predictive
efficacy of the architectures (gross returns). To eval-
uate resilience to trading frictions without relying on
arbitrary assumptions, we conduct a post-hoc, asset-
level breakeven transaction cost analysis. For each
asset, we compute the breakeven cost c∗k, which rep-
resents the maximum constant friction that specific
asset’s strategy can endure before its cumulative PnL
is driven to zero. The formal mathematical definition
of c∗kand the comprehensive per-asset breakeven re-
sults are detailed in Appendix E. As detailed by [25],
we ensemble the positions of the top S seeds based
on validation loss to reduce turnover and improve ro-
bustness to transactions costs.

#### 2.2 Linear Baselines

We include a set of linear models as classical and
modern baselines to contextualize the performance
of deep architectures. The mathematical background
can be found in B.

Autoregressive Model (AR1x). The AR1x
model [17] serves as a minimal temporal benchmark,
capturing short-term autocorrelation in returns. This
model applies AR(1) per feature, since the input
has multiple features. Its performance provides a
lower bound on the benefit of incorporating temporal
context and highlights whether short-horizon depen-
dence alone is sufficient for profitable trading.

DLinear and NLinear. DLinear and NLinear
[15] are non-recurrent linear models that apply
learned linear mappings to fixed-length input win-
dows. DLinear explicitly decomposes the input into
trend and seasonal components, while NLinear oper-
ates on normalized inputs. These models have shown
strong performance on data with pronounced linear
structure or seasonality.

#### 2.3 Transformer-Based Architectures

#### Without Explicit Recurrence

```
Within the Transformer B.2 temporal context, i.e.,
context window, the model learns relative temporal
importance. However, it does not explicitly encode
a temporal state representation and is therefore sus-
ceptible to overfitting to outliers.
Transformer-based models are known to perform
well in many tasks, but often struggle with finan-
cial time-series forecasting [26, 27]. In the implemen-
tation of the following models, we use parallel off-
set streams so we obtain dense per-timestep outputs,
similar to the output of the other models.
```

```
iTransformer. The inverted Transformer [3] ap-
plies attention across feature dimensions rather than
time, treating each feature as a token. While this de-
sign improves parameter efficiency, it removes explicit
temporal recurrence and relies solely on attention to
capture dynamics.
```

```
PatchTST. PatchTST [4] segments the input se-
quence into temporal patches, which are embedded
and processed via self-attention. Representing the
data in patches inherently smooths the financial time-
series and has shown to improve the performance in
long-term forecasting [28]. The receptive field is in-
creased to fully exploit the temporal context captured
by the patch-based architecture.
```

#### 2.4 State-Space and Implicitly Recur-

#### rent Models

```
Mamba and Mamba2. Mamba models [6, 29]
belong to the class of selective state-space models
(SSMs) B.3, which maintain a latent state that is up-
dated recursively over time. Unlike attention-based
architectures, SSMs provide an implicit temporal re-
currence with linear-time complexity, making them
well-suited for long sequences and noisy environments
such as financial time series.
At a high level, these models update a hidden state,
often based on High-order Polynomial Projection Op-
erators (HiPPO) [8], which summarizes past informa-
tion and produces outputs conditioned on the current
```

state. The parameters governing the state evolution
are dynamically modulated by neural networks con-
ditioned on the input, enabling adaptive temporal
dynamics while preserving computational efficiency.
Mamba2 refines this formulation by simplifying the
state transition structure and increasing head dimen-
sionality, leading to improved numerical stability and
throughput. In our implementation, we use a static
HiPPO-based state transition matrix with a fixed
horizon, rather than per-step adaptive horizon jit-
ter. This design choice improves noise tolerance [30]
and stabilizes learning in the presence of heavy-tailed
returns and regime shifts, which are common in finan-
cial markets.

#### 2.5 Recurrent Models

LSTM. Long Short-Term Memory (LSTM) B.
networks maintain an explicit recurrent state, con-
sisting of a hidden state and a memory cell, that is
updated sequentially over time, enabling the model
to capture temporal dependencies, but with an expo-
nentially decaying long-horizon temporal state [11].
This architecture has proven to be useful in many
cases in finance [18].

xLSTM. xLSTM [12] extends the classical LSTM
by introducing exponential gating and stabilized
memory normalization to improve long-range infor-
mation retention and gradient flow B.4. While stan-
dard LSTMs rely on sigmoid gates that may saturate
and induce premature forgetting, xLSTM replaces
these with exponentiated gate activations followed
by normalization, yielding approximately linear be-
havior over a wider dynamic range. This modifica-
tion mitigates vanishing memory effects and allows
the model to retain rare but economically meaning-
ful signals.
xLSTM comprises two variants: scalar LSTM
(sLSTM), which maintains a scalar memory state
updated via normalized exponential gates, and ma-
trix LSTM (mLSTM), which generalizes the mem-
ory state to a matrix-valued representation, en-
abling higher memory capacity and associative recall
through key–value storage mechanisms. This design

```
increases representational richness and improves scal-
ability compared to classical recurrent architectures.
From a financial perspective, the ability to preserve
temporally distant but informative signals and adap-
tively revise memory states is particularly relevant
in low signal-to-noise and regime-dependent environ-
ments [31].
```

```
Patch sLSTM (PsLSTM). Patch sLSTM [13] in-
tegrates the patching strategy of PatchTST with the
recurrent inductive bias of sLSTM. Given a multi-
variate time series of length L with d channels, each
channel is treated as an independent univariate se-
quence and segmented into non-overlapping temporal
patches:
```

```
̃x(pi)= Patch
```

######

```
x(ti:t)+ℓ− 1
```

######

```
, (i = 1,...,d) (12)
```

```
where ℓ denotes the patch length.
Each patch embedding ̃x
(i)
p is then processed by an
sLSTM, withsharedparametersacrosschannels:
```

```
h(pi)= sLSTM(h(pi−) 1 , ̃x(pi)). (13)
```

```
This design preserves channel independence while en-
forcing parameter sharing, preventing premature fea-
ture mixing and improving generalization.
By operating at the patch level, PsLSTM reduces
sensitivity to high-frequency noise and allows the re-
current mechanism to focus on medium-term tem-
poral structure. The exponential gating of sLSTM
further enhances memory persistence across patches,
enabling the model to capture regime-level dynamics
and rare events. After recurrent processing, hidden
states across channels are flattened and projected to
form the final prediction.
Patch sLSTM thus combines the noise robustness
and efficiency of patch-based modeling with the long-
range memory advantages of exponential-gated recur-
rence, which may be advantageous in financial time
series characterized by non-stationary and bad signal-
to-noise ratio.
```

#### 2.6 Hybrids

```
Several hybrid architectures are considered to im-
prove robustness in financial time series by enhanc-
```

ing the signal-to-noise ratio and stabilizing temporal
state updates B.5.
Variable Selection Networks (VSNs) B.5, inspired
by the Temporal Fusion Transformer, are used to per-
form feature-wise nonlinear embedding and dynamic
soft selection of relevant covariates at each time step.
This mechanism adaptively suppresses noisy or unin-
formative features.
Another strategy to improve robustness is the in-
clusion of an LSTM-based temporal encoder prior
to the main model. By explicitly maintaining a re-
current state, this preprocessing stage filters high-
frequency noise and stabilizes downstream represen-
tations.

VSN+LSTM (VLSTM). VLSTM combines a
VSN with an LSTM encoder to construct a com-
pact temporal state representation. This was the core
component of the X-Trend architecture used for con-
structing sequence representations for few-shot learn-
ing in financial time series [32]. The VSN produces
dynamically weighted feature embeddings, which are
then processed sequentially by an LSTM to aggregate
long-range temporal information.

VSN–Mamba2. This hybrid augments Mamba
with a VSN to separate feature selection from tempo-
ral modeling. The VSN filters noisy covariates before
passing the selected representation to the recurrent
state-space model, improving robustness in noisy fi-
nancial environments.

LSTM + PatchTST (LPatchTST). This ar-
chitecture combines explicit recurrence with atten-
tion by using an LSTM as a channel-wise temporal
denoiser prior to PatchTST. The LSTM stabilizes
per-channel representations, while PatchTST aggre-
gates medium- and long-range dependencies across
denoised temporal patches.

VSN + xLSTM (VxLSTM). In this hybrid,
VSN-selected representations are directly fed into an
xLSTM. The matrix-valued memory of xLSTM en-
ables the model to capture higher-order temporal in-

```
teractions and long-range dependencies beyond the
capacity of vector-based recurrent architectures.
```

#### 2.7 Complete Structured Model

```
Temporal Fusion Transformer (TFT). TFT [5]
integrates gated recurrent layers with interpretable
attention mechanisms. A recurrent encoder captures
local temporal dynamics, where LSTM is used as the
recurrent encoder, while multi-head attention aggre-
gates information across time:
```

```
ˆty = Attn(LSTM(x1:t)). (14)
```

```
Variable selection networks, static covariate encoders,
and gating mechanisms further improve robustness,
making TFT a strong benchmark for time-series fore-
casting and specifically financial forecasting [33].
```

## 3 Empirical Results

```
This section presents a comprehensive evaluation
of the out-of-sample performance of the consid-
ered models across multiple market regimes, perfor-
mance metrics, and computational dimensions, Ap-
pendix D.1. We focus on (i) risk-adjusted returns
across subperiods, (ii) aggregate return performance
and statistical significance, (iii) downside and tail-
risk characteristics, (iv) robustness to seed selec-
tion and experimental budget, and (v) the trade-off
between predictive performance and computational
complexity.
```

#### 3.1 Data Description

```
Our empirical analysis was conducted on a diversified
cross-asset futures and currency dataset [34]. The
futures data comprises of instruments from five as-
set classes: bonds, commodities, energy, foreign ex-
change, and equity indices. Daily closing prices were
used to construct returns and predictive features. For
futures, we use continuous contracts formed by link-
ing adjacent maturities using a ratio-adjusted back-
wards methodology (i.e., back-adjusted to remove
roll-induced price jumps). A detailed description of
```

data construction and exploratory analysis is pro-
vided in Appendix A.
The dataset [34] exhibits several well-documented
stylized facts of financial time series, including heavy-
tailed return distributions, volatility clustering, and
strong deviations from Gaussianity. These proper-
ties are illustrated formally in Appendix A, where we
reported distributional diagnostics (e.g., QQ-plots)
and volatility dynamics. As such, the dataset pro-
vides a realistic and challenging benchmark for eval-
uating nonlinear forecasting architectures in a cross-
asset setting.
Although the empirical evidence is robust within
our cross-asset benchmark, the results should be in-
terpreted as conditional on the specific dataset and
period considered. Extending the analysis to alterna-
tive markets and sampling frequencies would further
clarify the external validity of the findings.

#### 3.2 Performance Across Market

#### Regimes

Table 1 reports out-of-sample Sharpe ratios aggre-
gated over multiple overlapping horizons from 2010
to 2024, while Table 11 presents annual Sharpe ra-
tios. Taken together, these results enable an evalu-
ation of year-to-year variability and medium-horizon
robustness across distinct market regimes, including
the post-GFC recovery, the low-volatility expansion
of the mid-2010s, and the elevated-uncertainty envi-
ronment following 2020.
Several systematic patterns emerge.
First, deep nonlinear sequence models substan-
tially outperform linear benchmarks on most aggre-
gated horizons. While linear specifications such as
AR1x, ARnx, DLinear, and NLinear occasionally
achieve strong single-year Sharpe ratios—particularly
during high-volatility years such as 2020—their per-
formance is highly variable across time. Their long-
horizon averages over 2010–2025 remain materially
below those of the strongest nonlinear architectures.
This instability is consistent with the limited rep-
resentational flexibility of linear dynamics in envi-
ronments characterized by non-stationarity, regime
shifts, and low signal-to-noise ratios.

```
In contrast, gated recurrent and hybrid sequence
models exhibit both higher average Sharpe ra-
tios and greater intertemporal consistency. The
LSTM already delivers strong performance (1.48 over
2010–2025), but its variance across years remains
non-negligible. Enhanced recurrent architectures
improve further. In particular, VLSTM achieves
a 2010–2025 Sharpe ratio of 2.40 and maintains
strong performance across subperiods, including 2.
over 2015–2025 and 1.88 over 2020–2025. Similarly,
LPatchTST achieves 2.31 over 2010–2025 and re-
mains stable across all aggregated windows, including
2.34 in the post-2020 regime. The Temporal Fusion
Transformer (TFT) also demonstrates robust per-
formance, with 2.27 over 2010–2025 and consistent
strength across medium-horizon splits.
These results suggest that architectures combin-
ing adaptive gating, representation compression, and
structured temporal abstraction are better suited to
financial data than either purely linear models or
attention-only baselines. The year-by-year break-
down further reveals that top-performing models
rarely collapse entirely in adverse years; rather, per-
formance degrades moderately while remaining eco-
nomically meaningful. This robustness is particularly
visible during volatile periods such as 2020–2022.
State-space models such as Mamba and Mamba
display more heterogeneous behavior. While certain
years exhibit strong Sharpe ratios (notably 2020 and
2022 for Mamba2), their aggregated performance re-
mains moderate (0.78 and 0.64 over 2010–2025 for
Mamba2 and Mamba, respectively). Augmenting
Mamba2 with a Variable Selection Network improves
medium-horizon averages (1.10 over 2010–2025), in-
dicating that explicit feature conditioning partially
mitigates instability, though it does not close the gap
to the strongest recurrent or hybrid models.
Transformer-based patching approaches show
mixed results. PatchTST achieves moderate long-run
averages (0.76 over 2010–2025), but exhibits higher
sensitivity to specific years. In contrast, LPatchTST,
which augments patching with stronger sequence
modeling, delivers consistently superior and more sta-
ble results, suggesting that patch segmentation alone
is insufficient without robust temporal state encod-
ing.
```

```
Table 1: Out-of-sample Sharpe Ratio by subperiod. Annual Sharpe Ratio in Table 11
```

### Strategy 2010-2025 2015-2025 2010-2015 2015-2020 2020-

### AR1x 0.77 0.70 0.74 0.06 1.

### ARnx 0.63 0.55 0.70 -0.01 1.

### DLinear 0.64 0.64 0.60 0.00 1.

### LSTM 1.48 1.33 1.83 1.60 1.

### VLSTM 2.40 2.25 2.57 2.61 1.

### Mamba2 0.78 0.86 0.54 0.18 1.

### VSN+Mamba2 1.10 1.14 0.95 0.54 1.

### PatchTST 0.76 0.80 0.59 0.57 1.

### LPatchTST 2.31 2.22 2.33 2.11 2.

### PsLSTM 1.74 1.74 1.60 1.84 1.

### TFT 2.27 2.08 2.47 2.08 2.

### VxLSTM 1.69 1.61 1.56 1.48 1.

### xLSTM 1.79 1.84 1.46 1.68 1.

### iTransformer 0.38 0.28 0.60 0.06 0.

### Mamba 0.64 0.28 0.51 -0.01 0.

### NLinear 0.66 0.68 0.60 0.14 1.

Finally, xLSTM-based architectures demonstrate a
compelling balance between performance and stabil-
ity. The xLSTM achieves a Sharpe ratio of 1.79 over
2010–2025, improving to 1.99 in the 2020–2025 pe-
riod. VxLSTM yields comparable results (1.69 over
2010–2025), while PsLSTM achieves 1.74. Impor-
tantly, these models maintain Sharpe ratios near or
above 1.5 across most aggregated horizons, indicat-
ing resilience to changing volatility regimes. Their
year-level profiles show fewer extreme drawdowns rel-
ative to classical LSTM, consistent with the hypoth-
esis that enriched state representations and alterna-
tive gating mechanisms enhance adaptability in non-
stationary environments.

Taken together, the annual and aggregated re-
sults reinforce the central hypothesis of the paper:
successful financial forecasting architectures bene-
fit from adaptive memory mechanisms, representa-
tion compression, and temporally stable state evo-

```
lution. Models that incorporate structured gating
and persistent state representations dominate both
linear baselines and generic state-space formulations
across nearly all evaluation horizons. As emphasized
throughout, these conclusions remain conditional on
the dataset and evaluation framework considered;
nevertheless, the consistency across multiple tempo-
ral aggregations provides evidence that the observed
performance differentials are not driven solely by iso-
lated years or singular market events.
```

#### 3.3 Aggregate Return Performance

#### and Statistical Significance

```
Table 2 reports full-sample out-of-sample perfor-
mance under a volatility-targeting constraint of
10%, presenting compound annual growth rates
(CAGR) [35], annualized returns (Ann. Ret.),
Sharpe Ratio (SR) [36], heteroskedasticity and au-
```

tocorrelation consistent t-statistics (t HAC) [37], hit
rate (Hit) [38], turnover, turnover as a multiple of
gross market value (xGMV) [23], and additional di-
agnostics relative to a passive long-only benchmark:
the information ratio (Info. Ratio), HAC t-statistic
relative to passive (t HAC v Passive), and correla-
tion with passive returns (Corr. v Passive). Collec-
tively, these measures capture economic magnitude,
statistical reliability, trading intensity, and incremen-
tal value relative to buy-and-hold exposure (see Ap-
pendix D), see Appendix D.
VLSTM delivers the strongest overall performance
within this framework. It achieves a 23.9% annu-
alized return with a Sharpe ratio of 2.39, exceed-
ing both linear and alternative deep learning bench-
marks. Its HAC-adjusted t-statistic of 8.81 indicates
high statistical reliability under heteroskedasticity-
and autocorrelation-consistent inference. The hit
rate of 58.8% suggests persistent directional accuracy.
Relative to the passive benchmark, VLSTM attains
an information ratio of 0.854 and an HAC t-statistic
of 3.31, indicating statistically distinguishable excess
performance. Its correlation with passive returns
(0.404) implies partial independence from broad mar-
ket exposure and meaningful diversification potential.
The hybrid LPatchTST model achieves compara-
ble economic performance, with a Sharpe ratio of 2.
and a CAGR of 25.5%. Passive-relative metrics re-
main elevated (information ratio 0.707; t HAC 2.75),
though modestly below VLSTM. Similarly, the Tem-
poral Fusion Transformer (TFT) delivers strong ab-
solute and relative performance (Sharpe 2.20), rein-
forcing the importance of structured sequence repre-
sentations and adaptive gating mechanisms. Across
these leading architectures, elevated Sharpe ratios
coincide with statistically significant passive-relative
improvements, suggesting that gains are not merely
attributable to implicit market timing or leverage ef-
fects under volatility targeting.
LSTM-based variants, including PsLSTM, VxL-
STM, and xLSTM, substantially outperform linear
baselines. Notably, xLSTM achieves a Sharpe ratio of
1.80 with a comparatively moderate turnover (482),
resulting in one of the strongest passive-relative diag-
nostics (information ratio 0.798; t HAC 2.90). This
combination of competitive returns and reduced trad-

```
ing intensity suggests improved efficiency in signal
extraction relative to classical LSTM, which requires
nearly double the turnover to achieve similar Sharpe
ratios. These findings are consistent with the view
that enriched state representations can improve the
signal-to-trade ratio in noisy financial environments.
Linear benchmarks (AR1x, ARnx, DLinear) ex-
hibit Sharpe ratios below one and comparatively
small HAC t-statistics. Passive-relative metrics are
near zero or negative, and correlations with the buy-
and-hold benchmark remain moderate. While these
models occasionally benefit from favorable return
persistence, their aggregate performance indicates
limited capacity to extract stable predictive structure
under non-stationarity.
State-space models display heterogeneous be-
havior. Mamba2 reduces trading intensity rela-
tive to most deep sequence models (turnover 233)
but achieves only moderate economic performance
(Sharpe 0.62). Augmenting with a Variable Selec-
tion Network improves both Sharpe ratio (0.97) and
passive-relative metrics, indicating that explicit fea-
ture conditioning enhances stability. Nevertheless,
their aggregate performance remains below that of
recurrent and hybrid architectures.
The inclusion of iTransformer provides an informa-
tive contrast. It exhibits by far the lowest turnover
(36) and xGMV (9.2), indicating a highly conserva-
tive trading profile with minimal portfolio rebalanc-
ing. However, this low implementation intensity coin-
cides with weak economic performance (Sharpe 0.35)
and statistically insignificant passive-relative diag-
nostics. This pattern suggests that extreme turnover
reduction may reflect under-reactivity to evolving re-
turn signals in non-stationary markets. In the present
setting, reduced trading alone does not generate eco-
nomic value; rather, successful models appear to bal-
ance adaptive responsiveness with controlled trading
intensity.
Figure 2 visually corroborates these quantitative
findings. Leading sequence-based models dominate
cumulative PnL trajectories while maintaining rel-
atively smooth performance paths, indicating that
elevated Sharpe ratios reflect persistent incremen-
tal returns rather than isolated return episodes. As
throughout, these findings remain conditional on the
```

Table 2: 2010–2025 Gross return performance, statistical significance, and passive-relative diagnostics (volatility-
targeted at 10%).

```
Model CAGR Ann. Ret. SR t (HAC) Hit Turnover xGMV Info. Ratio t (HAC) v Passive Corr. v Passive
Passive 0.0435 0.0476 0.48 1.65 0.531 – – – – –
AR1x 0.0813 0.0831 0.83 3.12 0.539 353.64 90.421 -0.0086 -0.0305 0.
ARnx 0.0646 0.0677 0.68 2.52 0.538 280.66 69.525 -0.0829 -0.3011 0.
DLinear 0.0750 0.0773 0.77 2.87 0.539 278.41 75.282 0.0141 0.0501 0.
LSTM 0.1351 0.1318 1.32 4.56 0.554 948.08 225.769 -0.0637 -0.2303 0.
VLSTM 0.2632 0.2388 2.39 8.81 0.588 966.86 218.369 0.8539 3.3071 0.
Mamba2 0.0587 0.0620 0.62 2.31 0.546 233.00 58.164 -0.0901 -0.3246 0.
VSN+Mamba2 0.0967 0.0973 0.97 3.65 0.555 329.11 78.842 0.1091 0.3936 0.
PatchTST 0.0847 0.0864 0.86 3.29 0.541 623.88 198.021 -0.2149 -0.7848 0.
LPatchTST 0.2550 0.2323 2.32 8.81 0.577 959.89 211.514 0.7070 2.7470 0.
PsLSTM 0.1868 0.1763 1.76 6.83 0.563 823.07 185.496 0.3981 1.5410 0.
TFT 0.2398 0.2201 2.20 8.13 0.584 912.81 223.231 0.6665 2.5487 0.
VxLSTM 0.1937 0.1821 1.82 6.89 0.574 775.88 159.438 0.4666 1.6727 0.
xLSTM 0.1937 0.1796 1.80 6.85 0.568 482.62 91.924 0.7984 2.9042 0.
iTransformer 0.0308 0.0353 0.35 1.26 0.529 36.32 9.203 -0.1539 -0.5563 0.
```

dataset and backtesting design considered.

#### 3.4 Downside Risk and Tail Behavior

While strong average performance is economically
relevant, robustness to adverse market conditions is
particularly important in financial applications. Ta-
ble 3 reported downside and tail-risk metrics, includ-
ing maximum drawdown (Max DD), Calmar ratio
(Calmar), worst three-month Sharpe ratio (Worst 3m
Sharpe), minimum annual Sharpe ratio (Min Ann.
Sharpe), and 5% conditional value at risk (CVaR
5%).
VLSTM and LPatchTST exhibited comparatively
moderate drawdowns alongside relatively high Cal-
mar ratios (1.15 and 1.47, respectively). VLSTM
achieved a worst three-month Sharpe ratio of -3.
and the lowest CVaR among the evaluated mod-
els. These results were consistent with comparatively
milder tail losses within the sample period.
The smallest maximum drawdown was observed
for VxLSTM (-11.8%), accompanied by the high-
est Calmar ratio (1.64). However, this configuration
generated lower average returns relative to VLSTM
in the aggregate performance analysis, suggesting a
more conservative return profile within the evaluation
framework.
In contrast, standard LSTM and PatchTST archi-
tectures experienced larger drawdowns and weaker
worst-period performance. This pattern indicated

```
greater sensitivity to extreme market movements dur-
ing the sample period.
Overall, VLSTM combined comparatively strong
average performance with moderate downside risk
measures. However, LPatchTST and xLSTM were
the most robust and were able to keep a favorable tail
behavior. Within the dataset considered, this balance
suggested a favorable trade-off between return gener-
ation and tail-risk exposure. As throughout, these
findings should be interpreted as conditional on the
sample period and backtesting design.
```

#### 3.5 Breakeven Transaction Cost

```
Breakeven transaction cost analysis, Appendix E, re-
veals substantial cross-asset heterogeneity. The ta-
bles report annualised, volatility-rescaled gross and
net returns together with annualised turnover and the
implied breakeven transaction cost c∗in basis points.
For VLSTM, several agricultural contracts (e.g.,
Lumber, Oats elec, and Milk III) exhibit high gross
returns but also relatively large breakeven costs (ex-
ceeding 20 bps). These contracts are compara-
tively illiquid therefore they are expected to have
high transaction costs and scalability is limited de-
spite their strong gross profitability. A broad mid-
dle group demonstrates moderate profitability, with
breakeven costs in the range of 5–10 bps. At the
lower end, high-turnover contracts (e.g., US 2Y Note
Composite Bond and Euro Schatz Bond) display very
```

```
Figure 2: Performance comparison across models 10% volatility-rescaled gross PnL.
```

small breakeven costs, indicating that profitability
is quickly eroded by transaction costs. These are
amongst the most liquid contracts; therefore, it is ex-
pected that they have tight spreads. However, they
can also be traded in high volumes. Finally, a small
subset of contracts generates negative gross returns,
resulting in negative c∗.

For xLSTM, the strongest contracts again in-
clude Lumber, Oats elec, Milk III, with even higher
breakeven costs for some assets (e.g., Lumber at 33.
bps), once again, noting that these are highly illiq-
uid contracts. Notably, xLSTM achieves materially
lower turnover for several equity and bond contracts
(e.g., ES, ZN), leading to higher breakeven margins
despite similar gross returns. However, the lower tail
contains more negative gross performers relative to
VLSTM.

Overall, xLSTM appears more transaction-cost ef-
ficient in several liquid contracts due to reduced
turnover, while VLSTM delivers broader cross-
sectional profitability.

#### 3.6 Robustness to Seed Selection and

#### Experimental Budget

```
Table 4 evaluated the robustness of the main find-
ings to a substantially reduced experimental bud-
get. Whereas the primary results in Sections 3.
and 3.4 were obtained by averaging performance over
the top 10 seeds, based on validation loss, from 50
independent runs, this table reported results using
only 25 random seeds and averaging over the top 5
realizations. This configuration represented a nois-
ier and more resource-constrained evaluation regime,
intended to assess whether model rankings and eco-
nomic patterns persisted under weaker aggregation.
Across models, performance levels and relative
ordering remained broadly stable. VLSTM again
achieved the highest absolute and risk-adjusted re-
turns, with a Sharpe ratio of 2.40 and an HAC-
adjusted t-statistic of 8.86, closely aligned with its
full-budget estimates. The TFT and VxLSTM also
maintained Sharpe ratios above 1.9 with statisti-
cally distinguishable HAC t-statistics, while the lin-
ear AR1x benchmark continued to exhibit compara-
tively weak performance across return metrics. This
```

```
Table 3: Downside risk and tail behavior (gross returns).
```

##### Model Max DD Calmar Worst 3m Sharpe Min Ann. Sharpe CVaR 5%

##### AR1x -0.167 0.49 -3.92 -0.59 0.

##### ARnx -0.206 0.31 -4.52 -0.90 0.

##### DLinear -0.180 0.42 -4.93 -0.94 0.

##### LSTM -0.342 0.40 -5.15 -1.51 0.

##### VLSTM -0.229 1.15 -3.68 -0.10 0.

##### Mamba2 -0.263 0.22 -4.06 -0.71 0.

##### VSN+Mamba2 -0.163 0.59 -4.00 -0.63 0.

##### PatchTST -0.176 0.48 -5.58 -1.21 0.

##### LPatchTST -0.174 1.47 -3.91 0.51 0.

##### PsLSTM -0.131 1.43 -3.80 -0.40 0.

##### TFT -0.232 1.03 -3.87 -0.14 0.

##### VxLSTM -0.118 1.64 -3.70 -1.31 0.

##### xLSTM -0.141 1.35 -3.57 -0.28 0.

##### Passive -0.308 0.14 -6.11 -1.53 0.

##### iTransformer -0.264 0.12 -3.93 -1.16 0.

stability in relative rankings was consistent with the
interpretation that performance differences were not
solely driven by extensive seed exploration or aggres-
sive averaging.
Downside and tail-risk characteristics exhibited
similar patterns. Maximum drawdowns, Calmar ra-
tios, and worst three-month returns remained quan-
titatively comparable to those observed under the
full-budget evaluation. VLSTM and VxLSTM main-
tained elevated Calmar ratios (1.66 and 1.88, respec-
tively), indicating a similar balance between return
generation and drawdown magnitude as in the pri-
mary specification. The 5% Conditional Value-at-
Risk (CVaR) did not materially deteriorate under re-
duced seed averaging, suggesting that tail-risk char-
acteristics were not highly sensitive to seed selection
within the tested range.
Overall, the reduced-seed experiment suggested
that both economic performance and downside risk
measures were reasonably stable to substantial re-
ductions in experimental budget. The persistence of
model rankings and risk profiles under noisier evalua-
tion conditions was consistent with the view that the

```
observed performance differences reflected structural
properties of the architectures within this bench-
mark, rather than purely favorable random initializa-
tion or extreme-seed selection. As throughout, these
conclusions remain conditional on the specific dataset
and backtesting protocol employed.
```

#### 3.7 Discussion

```
The empirical results presented across return, risk,
tail, cost, and robustness diagnostics suggest sev-
eral consistent patterns regarding the inductive bi-
ases required for successful financial time-series mod-
eling. As emphasized throughout, these interpreta-
tions remain conditional on the dataset and eval-
uation framework considered, as discussed in Sec-
tion 3.1.
The dataset exhibits non-stationarity, heavy tails,
low signal-to-noise ratios, and pronounced deviations
from Gaussianity – stylized facts widely documented
in financial return series. To the extent that other fi-
nancial time series share these characteristics, similar
architectural considerations may apply.
```

```
Table 4: Reduced-seed benchmark (25 runs, top 5 seeds selected).
```

```
Return Performance Risk and Downside Metrics
Model CAGR Ann. Ret. Sharpe t (HAC) Hit Model Max DD Calmar Worst 3m Min Ann. CVaR 5%
AR1 0.084 0.085 0.854 3.208 0.537 AR1 -0.165 0.506 -4.068 -0.524 0.
VLSTM 0.264 0.240 2.397 8.857 0.589 VLSTM -0.159 1.664 -3.730 -0.250 0.
TFT 0.251 0.229 2.290 8.575 0.586 TFT -0.210 1.196 -3.499 -0.175 0.
VxLSTM 0.203 0.190 1.898 7.236 0.578 VxLSTM -0.108 1.878 -3.718 -1.064 0.
```

First, architectures maintaining explicit recur-
rent state representations consistently outperformed
purely attention-based models across most perfor-
mance dimensions. This advantage extends beyond
average Sharpe ratios to include downside protection
and tail behavior. Models such as VLSTM, VxL-
STM, and LPatchTST not only achieved strong full-
sample Sharpe ratios, but also demonstrated superior
Calmar ratios and materially higher minimum annual
Sharpe ratios. In particular, LPatchTST exhibited
an exceptionally stable worst-year profile, a property
of practical relevance in institutional settings where
drawdown control and avoidance of severely nega-
tive years are often prioritized over marginal improve-
ments in average Sharpe.
Second, robustness must be evaluated multidimen-
sionally. While VLSTM achieved the highest over-
all Sharpe ratio and strong passive-relative perfor-
mance, VxLSTM and LPatchTST exhibited superior
downside-adjusted characteristics in certain metrics,
including Calmar ratio and minimum annual Sharpe.
This distinction highlights an economically impor-
tant trade-off: maximizing mean risk-adjusted return
does not necessarily coincide with minimizing draw-
down severity or tail exposure. The choice of archi-
tecture may therefore depend on investor preference
over the mean–tail trade-off.
Third, turnover and transaction cost robustness
meaningfully differentiate models. Breakeven trans-
action cost analysis reveals that xLSTM achieves the
highest portfolio-level cost buffer, exceeding VLSTM
despite slightly lower average Sharpe. This indi-
cates greater resilience to implementation frictions
and a higher signal-to-trade efficiency. Conversely,
extremely low-turnover models such as iTransformer
exhibit weak predictive performance, suggesting that
insufficient responsiveness to evolving signals can be

```
as detrimental as excessive trading. Successful archi-
tectures appear to strike a balance between adaptive
state updating and controlled portfolio rebalancing.
```

```
Fourth, robustness to seed selection and experi-
mental budget strengthens the credibility of the main
findings. Performance rankings remain largely pre-
served under reduced seed aggregation, indicating
that the reported differences are not artifacts of favor-
able initialization. This stability is particularly im-
portant in low signal-to-noise financial environments,
where variance across runs can otherwise confound
interpretation.
```

```
Fifth, state-space models and purely linear bench-
marks exhibit heterogeneous and regime-sensitive be-
havior. While linear models occasionally perform
well in high-volatility subperiods, they fail to deliver
consistent cross-metric robustness. State-space ar-
chitectures offer computational efficiency but require
additional structure—such as feature selection or en-
riched state dynamics—to achieve competitive eco-
nomic performance.
```

```
Taken together, the evidence suggests – though
does not definitively establish – that successful finan-
cial forecasting architectures benefit from: (i) persis-
tent and adaptively gated state representations, (ii)
mechanisms for representation compression or feature
conditioning, (iii) controlled trading intensity that
preserves implementation robustness, and (iv) stabil-
ity under adverse market realizations. Importantly,
the most economically attractive models are not nec-
essarily those with the highest average Sharpe, but
those that jointly balance return, drawdown control,
tail resilience, and transaction cost tolerance within
realistic deployment constraints.
```

#### 3.8 Computational Efficiency and

#### Model Complexity

Table 5 reported parameter counts as well as asymp-
totic runtime and memory complexities across mod-
els. The final number of trainable parameters and the
corresponding hyperparameter ranges are provided in
Appendix C. Linear models exhibited minimal com-
putational costs in terms of both parameterization
and asymptotic complexity, although their empirical
performance in previous subsections was compara-
tively weaker within this benchmark.
Among nonlinear architectures, VLSTM combined
competitive empirical performance with linear mem-
ory complexity in sequence length and without
quadratic attention terms. Relative to transformer-
based models such as PatchTST or TFT, this resulted
in lower asymptotic memory growth with respect to
sequence length. Mamba and Mamba2 offered com-
parable asymptotic efficiency, though their empirical
performance in this study was more heterogeneous
across evaluation metrics.
Hybrid architectures, including LPatchTST and
VSN-enhanced variants, involved higher parame-
ter counts and additional computational overhead.
Within the present evaluation, these increases in
complexity were not uniformly associated with pro-
portional performance improvements. This pattern
suggested that asymptotic complexity alone did not
determine empirical effectiveness, and that architec-
tural inductive bias may play an important role in
financial time-series modeling. As with all preced-
ing results, these observations are conditional on the
specific dataset, sequence lengths, and implementa-
tion choices considered.

## 4 Conclusions

This paper presented a comprehensive benchmark of
deep learning architectures for financial time-series
prediction under a unified experimental framework
spanning 15 years, multiple asset classes, heteroge-
neous market regimes, and a broad set of economic
and statistical diagnostics. Models were evaluated
not only on average risk-adjusted returns, but also

```
on downside risk, tail exposure, transaction cost ro-
bustness, implementation intensity, and sensitivity to
random initialization.
Several robust patterns emerged:
First, purely linear models exhibited occasional
regime-specific competitiveness but failed to de-
liver stable multi-metric performance. Their lim-
ited adaptability to non-stationarity and structural
change constrained their long-horizon effectiveness.
Second, architectures explicitly designed to learn
structured and adaptively gated temporal represen-
tations consistently outperformed generic attention-
based and state-space alternatives. VLSTM achieved
the highest overall Sharpe ratio and strong passive-
relative performance. However, models such as
VxLSTM and LPatchTST demonstrated superior
downside-adjusted robustness, including stronger
Calmar ratios and more stable worst-year outcomes.
These results highlight that mean risk-adjusted re-
turn and drawdown resilience need not coincide, and
that investor objectives may favor different architec-
tural trade-offs.
Third, transaction cost robustness meaningfully
differentiates models. xLSTM achieved the highest
breakeven cost buffer at the portfolio level, indicating
improved signal-to-trade efficiency. Extremely low-
turnover architectures, such as iTransformer, exhib-
ited limited predictive strength, suggesting that in-
sufficient responsiveness to evolving signals may un-
dermine economic performance. Effective models ap-
pear to balance adaptive state updating with disci-
plined trading intensity.
Fourth, performance rankings remained largely
stable under reduced seed aggregation and experi-
mental budgets, reinforcing that observed differences
are not artifacts of favorable initialization or exces-
sive tuning.
Collectively, the evidence suggests that success-
ful financial forecasting architectures benefit from
persistent and adaptively gated state representations,
representation compression or feature conditioning
mechanisms, and efficient translation of predictive
signals into implementable portfolio decisions. Im-
portantly, the most economically attractive models
are those that jointly balance average return, draw-
down control, tail robustness, and implementation
```

```
Model Number of Parameters Runtime Complexity Memory Complexity
TFT O(CH^2 + ℓH^2 ) O(L^2 H + LCH^2 ) O(L^2 + LH)
VLSTM O(CH^2 + H^2 ) O(LCH^2 + LH^2 ) O(LH)
Mamba O(ℓH^2 ) O(LH^2 ) O(LH)
Mamba2 O(ℓH^2 ) O(LH^2 ) O(LH)
VSN + Mamba2 O(CH^2 + ℓH^2 ) O(LCH^2 + LH^2 ) O(LH)
AR1x O( 1 ) O(L) O( 1 )
LSTM O(H^2 + CH) O(LH^2 ) O(LH)
NLinear O(LC) O(LC) O(LC)
DLinear O(LC) O(LC) O(LC)
xLSTM O(H^2 + CH) O(LH^2 ) O(LH)
Patch sLSTM O(P CH + H^2 ) O(N H^2 ) O(N H)
VSN + xLSTM O(H^2 (C + H)) O(LH^2 ) O(LH^2 )
PatchTST O(P CH + ℓH^2 ) O(N^2 H) O(N^2 + N H)
LSTM + PatchTST O(H^2 + P CH + ℓH^2 ) O(LH^2 + N^2 H) O(LH + N^2 )
iTransformer O(ℓH^2 ) O(C^2 H + LCH) O(C^2 + LC)
```

Table 5: Comparison of parameter count, runtime complexity, and memory complexity for different time-series
forecasting models. The emphasis on order of results is as follows: best, second_best, third_best. The forecasting
horizon is fixed to one. L=sequence length, C=number of input features, H=hidden dimension, N=L− S=number
of patches (for shifted versions adjusted for each time step), P=patch length, S=stride, K=convolution kernel size,
ℓ=number of layers, M=number of attention heads.

feasibility.
The conclusions remain conditional on the dataset,
market universe, and backtesting assumptions em-
ployed. Nevertheless, by evaluating models under re-
alistic non-stationarity, heavy tails, volatility cluster-
ing, and transaction cost constraints, this benchmark
aims to reflect the statistical and economic challenges
inherent to practical financial forecasting. We hope it
provides a transparent empirical reference point for
future research and encourages architectural devel-
opment guided not only by computational considera-
tions, but also by the distinctive structural properties
of financial markets.

## Acknowledgments

Kieran Wood would like to thank the Oxford-Man In-
stitute of Quantitative Finance for its generous sup-
port.

## References

```
[1] Ashish Vaswani, Noam Shazeer, Niki Parmar,
Jakob Uszkoreit, Llion Jones, Aidan N. Gomez,
Łukasz Kaiser, and Illia Polosukhin. Attention is
all you need. InAdvancesinNeuralInformation
ProcessingSystems, volume 30, 2017.
```

```
[2] Saidul Islam, Hanae Elmekki, Ahmed Elsebai,
Jamal Bentahar, Najat Drawel, Gaith Rjoub,
and Witold Pedrycz. A comprehensive survey
on applications of transformers for deep learning
tasks. ExpertSystemswithApplications, 241:
122666, 2024. doi: 10.1016/j.eswa.2023.122666.
```

```
[3] Yong Liu, Tengge Hu, Haoran Zhang, Haixu
Wu, Shiyu Wang, Lintao Ma, and Ming-
sheng Long. iTransformer: Inverted trans-
formers are effective for time series forecast-
ing. InInternationalConferenceonLearning
Representations(ICLR), 2024. doi: 10.48550/
arXiv.2310.06625. URL https://openreview.
net/forum?id=JePfAI8fah.
```

```
[4] Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong,
```

```
and Jayant Kalagnanam. A time series is
worth 64 words: Long-term forecasting with
transformers. In International Conferenceon
Learning Representations (ICLR), 2023. doi:
10.48550/arXiv.2211.14730. URL https://
openreview.net/forum?id=Jbdc0vTOcol.
```

```
[5] Bryan Lim, Sercan Ö. Arık, Nicolas Loeff, and
Tomas Pfister. Temporal fusion transformers for
interpretable multi-horizon time series forecast-
ing. International Journalof Forecasting, 37
(4):1748–1764, 2021. doi: 10.1016/j.ijforecast.
2021.03.012. URL https://doi.org/10.1016/
j.ijforecast.2021.03.012.
```

```
[6] Tri Dao and Albert Gu. Transformers are
SSMs: Generalized models and efficient algo-
rithms through structured state space dual-
ity, 2024. URL https://arxiv.org/abs/2405.
21060.
```

```
[7] Angelos Katharopoulos, Apoorv Vyas, Niko-
laos Pappas, and François Fleuret. Transform-
ers are RNNs: Fast autoregressive transform-
ers with linear attention. In Proceedings of
the37thInternationalConferenceonMachine
Learning (ICML), volume 119 of Proceedings
ofMachineLearningResearch, 2020. doi: 10.
48550/arXiv.2006.16236. URL https://arxiv.
org/abs/2006.16236.
```

```
[8] Albert Gu, Tri Dao, Stefano Ermon, Atri
Rudra, and Christopher Ré. HiPPO: Recurrent
memory with optimal polynomial projections.
InAdvancesinNeuralInformationProcessing
Systems, volume 33, pages 1474–1487, 2020.
URL https://arxiv.org/abs/2008.07669.
```

```
[9] Leo Feng, Felix Tung, Mostofa O. Ahmed,
Yoshua Bengio, and Houd Hajimirsadeghi. Were
RNNs all we needed?, 2024. URL https://
arxiv.org/abs/2410.01201.
```

[10] Hansika Hewamalage, Christoph Bergmeir, and
Kasun Bandara. Recurrent neural networks
for time series forecasting: Current status and
future directions. International Journal of
Forecasting, 37(1):388–427, 2021. doi: 10.1016/

```
j.ijforecast.2020.06.008. URL https://arxiv.
org/abs/1909.00590.
```

```
[11] Sepp Hochreiter and Jürgen Schmidhuber. Long
short-term memory.NeuralComputation, 9(8):
1735–1780, 1997. doi: 10.1162/neco.1997.9.8.
1735.
```

```
[12] Maximilian Beck, Konstantin Pöppel, Markus
Spanring, Andreas Auer, Olga Prudnikova,
Michael Kopp, Günter Klambauer, Johannes
Brandstetter, and Sepp Hochreiter. xLSTM: Ex-
tended long short-term memory. InAdvances
inNeuralInformationProcessingSystems, vol-
ume 37, 2024. doi: 10.48550/arXiv.2405.04517.
URL https://arxiv.org/abs/2405.04517.
```

```
[13] Yaxuan Kong, Zepu Wang, Yuqi Nie, Tian
Zhou, Stefan Zohren, Yuxuan Liang, Peng Sun,
and Qingsong Wen. Unlocking the power
of LSTM for long term time series forecast-
ing. InProceedingsoftheAAAIConference
on Artificial Intelligence, volume 39, pages
11968–11976, 2025. doi: 10.1609/aaai.v39i11.
```

33303. URL https://ojs.aaai.org/index.
       php/AAAI/article/view/33303.

```
[14] Yuxuan Wang, Haixu Wu, Jiaxiang Dong, Yong
Liu, Chen Wang, Mingsheng Long, and Jian-
min Wang. Deep time series models: A com-
prehensive survey and benchmark, 2024. URL
https://arxiv.org/abs/2407.13278.
```

```
[15] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang
Xu. Are transformers effective for time se-
ries forecasting? InProceedingsoftheAAAI
ConferenceonArtificialIntelligence, volume 37,
pages 11121–11128, 2023. doi: 10.1609/aaai.
v37i9.26317. URL https://ojs.aaai.org/
index.php/AAAI/article/view/26317.
```

```
[16] Rama Cont. Empirical properties of asset
returns: Stylized facts and statistical issues.
QuantitativeFinance, 1(2):223–236, 2001. doi:
10.1080/713665670.
```

```
[17] George Udny Yule. On a method of inves-
tigating periodicities in disturbed series, with
```

```
special reference to wolfer’s sunspot numbers.
PhilosophicalTransactionsoftheRoyalSociety
of London.SeriesA, 226:267–298, 1927. doi:
10.1098/rsta.1927.0007. URL https://www.
jstor.org/stable/91170.
```

[18] Omer Berat Sezer, Mehmet Ugur Gudelek, and
Ahmet Mert Ozbayoglu. Financial time series
forecasting with deep learning: A systematic lit-
erature review: 2005–2019, 2019. URL https:
//arxiv.org/abs/1911.13288.

[19] Xinhe Liu and Wenmin Wang. Deep time se-
ries forecasting models: A comprehensive sur-
vey. Mathematics, 12(10):1504, 2024. doi: 10.
3390/math12101504. URL https://www.mdpi.
com/2227-7390/12/10/1504.

[20] Mateusz Buczyński, Marcin Chlebus, Katarzyna
Kopczewska, and Marcin Zajenkowski. Finan-
cial time series models—comprehensive review
of deep learning approaches and practical recom-
mendations.EngineeringProceedings, 39(1):79,

2023. doi: 10.3390/engproc2023039079. URL
      https://www.mdpi.com/2673-4591/39/1/79.

[21] Cheng Guo and Felix Berkhahn. Entity embed-
dings of categorical variables. arXivpreprint
arXiv:1604.06737, 2016. URL https://arxiv.
org/abs/1604.06737.

[22] Tobias J. Moskowitz, Yao Hua Ooi, and
Lasse Heje Pedersen. Time series momen-
tum. JournalofFinancialEconomics, 104(2):
228–250, 2012. doi: 10.1016/j.jfineco.2011.11.

3. URL https://www.sciencedirect.com/
   science/article/pii/S0304405X11002613.

[23] Campbell R. Harvey, Edward Hoyle, Rohit Ko-
rgaonkar, Sandy Rattray, Matthew Sargaison,
and Otto Van Hemert. The impact of volatility
targeting.TheJournalofPortfolioManagement,
45(1):14–33, 2018. doi: 10.3905/jpm.2018.45.1. 014.

[24] Bryan Lim, Stefan Zohren, and Stephen
Roberts. Enhancing time-series momentum
strategies using deep neural networks. The

```
JournalofFinancialDataScience, 1(4):19–38,
```

2019. doi: 10.3905/jfds.2019.1.015.

```
[25] Kieran Wood, Stephen J. Roberts, and Ste-
fan Zohren. DeePM: Regime-robust deep learn-
ing for systematic macro portfolio management,
```

2026. URL https://arxiv.org/abs/2601.
2027.

```
[26] Yufa Zhou, Yixiao Wang, Surbhi Goel, and
Anru R. Zhang. Why do transformers fail to
forecast time series in-context?, 2025. URL
https://arxiv.org/abs/2510.09776.
```

```
[27] Yong Liu, Haixu Wu, Jianmin Wang, and Ming-
sheng Long. Non-stationary transformers: Ex-
ploring the stationarity in time series fore-
casting, 2022. URL https://arxiv.org/abs/
2205.14415.
```

```
[28] José M. Oliveira and Pedro Ramos. Evaluating
the effectiveness of time series transformers for
demand forecasting. Mathematics, 12(17):2728,
```

2024. doi: 10.3390/math12172728. URL https:
      //www.mdpi.com/2227-7390/12/17/2728.

```
[29] Albert Gu and Tri Dao. Mamba: Linear-time se-
quence modeling with selective state spaces. In
ProceedingsoftheFirstConferenceonLanguage
Modeling(COLM), 2024. doi: 10.48550/arXiv.
2312.00752. URL https://openreview.net/
forum?id=tEYskw1VY2.
```

```
[30] Albert Gu, Karan Goel, and Christopher Ré.
Efficiently modeling long sequences with struc-
tured state spaces. InInternationalConference
onLearningRepresentations(ICLR), 2022. doi:
10.48550/arXiv.2111.00396. URL https://
openreview.net/forum?id=uYLFoz1vlAC.
```

```
[31] Carl Fjellström. Long short-term memory neu-
ral network for financial time series, 2022. URL
https://arxiv.org/abs/2201.08218.
```

```
[32] Kieran Wood, Samuel Kessler, Stephen J.
Roberts, and Stefan Zohren. Few-shot learn-
ing patterns in financial time series for trend-
following strategies. TheJournalofFinancial
```

```
DataScience, 6(2):88–115, 2024. doi: 10.3905/
jfds.2024.1.157.
```

[33] Kieran Wood, Sven Giegerich, Stephen Roberts,
and Stefan Zohren. Trading with the momentum
transformer: An intelligent and interpretable
architecture, 2021. URL https://arxiv.org/
abs/2112.08534.

[34] Pinnacle Data Corp. Pinnacle data corp.
CLC database. Website. URL https://
pinnacledata2.com/clc.html.

[35] Edwin J. Elton, Martin J. Gruber, Stephen J.
Brown, and William N. Goetzmann. Modern
PortfolioTheoryandInvestmentAnalysis. John
Wiley & Sons, 9 edition, 2014.

[36] William F. Sharpe. The sharpe ratio.Journalof
PortfolioManagement, 21(1):49–58, 1994.

[37] Whitney K. Newey and Kenneth D. West. A
simple, positive semi-definite, heteroskedasticity
and autocorrelation consistent covariance ma-
trix.Econometrica, 55(3):703–708, 1987.

[38] M. Hashem Pesaran and Allan Timmermann.
A simple nonparametric test of predictive per-
formance. Journal of Business & Economic
Statistics, 10(4):461–465, 1992.

[39] Cheng Zhang, Nilam Nur Amir Sjarif, and
Roslina Ibrahim. Deep learning models for price
forecasting of financial time series: A review of
recent advancements (2020–2022), 2023. URL
https://arxiv.org/abs/2304.11228.

[40] Andrea Bucci. Realized volatility fore-
casting with neural networks. Journal
of Financial Econometrics, 18(3):502–531,

2020. doi: 10.1093/jjfinec/nbaa008. URL
      https://academic.oup.com/jfec/article/
      18/3/502/5856840.

[41] Teuvo Kohonen. Correlation matrix memories.
IEEETransactionsonComputers, C-21(4):353–
359, 1972. doi: 10.1109/TC.1972.5008975.

```
[42] James A. Anderson. A simple neural network
generating an interactive memory.Mathematical
Biosciences, 14:197–220, 1972. doi: 10.1016/
0025-5564(72)90075-2.
```

```
[43] K. Nakano. Associatron—a model of associative
memory. IEEETransactionsonSystems,Man,
andCybernetics, SMC-2(3):380–388, 1972. doi:
10.1109/TSMC.1972.4309133.
```

```
[44] John R. Anderson, James W. Silverstein,
Steven A. Ritz, and Randall S. Jones. Distinc-
tive features, categorical perception, and prob-
ability learning: Some applications of a neu-
ral model.PsychologicalReview, 84(5):413–451,
```

1977. doi: 10.1037/0033-295X.84.5.413.

```
[45] Diederik P. Kingma and Jimmy Ba. Adam:
A method for stochastic optimization.
In International Conference on Learning
Representations, 2015.
```

## A Data Construction and Ex-

## ploratory Analysis

This appendix documents the construction of all vari-
ables used in the empirical analysis and provides de-
tailed evidence on their distributional properties.

#### A.1 Raw Data and Return Construc-

#### tion

The raw dataset consists of daily observations with
three fields: date, ticker, and closing price. From
these, daily returns are computed as

```
rt=
```

```
Pt− Pt− 1
Pt− 1
```

###### , (15)

where Ptdenotes the closing price at time t.

#### A.2 Volatility Estimation

Daily volatility is estimated using an exponentially
weighted moving average (EWMA) estimator. Let
λ = span^2 +1. The conditional mean and variance
evolve according to

```
μt= λrt+ (1− λ)μt− 1 , (16)
σ^2 t= λ(rt− μt)^2 + (1− λ)σ^2 t− 1. (17)
```

#### A.3 Distribution of Returns and

#### Volatility

Figure 3 reports the empirical distribution of daily
returns across all assets. The distribution is sharply
peaked around zero and exhibits pronounced lep-
tokurtosis. Figure 4 shows the distribution of realized
volatility on a logarithmic scale, highlighting strong
right skewness and a long upper tail.
Figure 5 presents a quantile–quantile plot against
the normal distribution together with the tail behav-
ior of absolute returns. Both figures indicate substan-
tial deviations from Gaussianity and slow tail decay.

#### A.4 Predictive Features

Normalized returns are constructed over multiple
horizons (1 day, 1 week, 1 month, 3 months, 6

```
Figure 3: Distribution of daily returns. To make the cen-
tral mass visible, the figure focuses on the bulk of the
distribution; tail behavior is examined separately.
```

```
Figure 4: Distribution of daily realized volatility (log
scale). Volatility exhibits strong right skewness and a
long upper tail.
```

```
months, and 1 year) as
```

```
rnormt,h =
```

```
rt,h
σt
```

###### √

```
h
```

###### , (18)

```
where h denotes the horizon in trading days.
```

```
Momentum indicators are further augmented us-
ing volatility-normalized and regime-adjusted Mov-
ing Average Convergence Divergence (MACD) sig-
```

Figure 5: Left: Quantile–quantile plot against the normal distribution. Right: Tail behavior of daily returns. The
figures indicate substantial deviations from Gaussianity and heavy-tailed return dynamics.

nals:

```
MACDt= EWMAh(Ts)(P )t− EWMAh(Tl)(P )t,
```

```
(19)
```

```
qt=
```

```
MACDt
Std 63 (P )t
```

###### , (20)

```
Signalt=
```

```
qt
Std 252 (q)t
```

###### . (21)

The empirical distributions of these features are
approximately symmetric and concentrated within
the interval [− 2 , 2], with nearly all observations con-
tained in [− 4 , 4], as shown in Figure 6.

#### A.5 Volatility-Scaled Exposure and

#### Target Variable

Volatility targeting induces a time-varying exposure
factor defined as

```
vs_factort=
```

###### 1

```
σt
```

###### . (22)

The empirical distribution of vs_factor exhibits
pronounced right skewness, Figure 7. A large share
of the mass is concentrated at relatively low values,

```
while a long right tail corresponds to regimes or con-
tracts characterized by unusually low realized volatil-
ity. These extreme realizations arise endogenously
from the volatility-scaling mechanism itself and do
not primarily reflect structural market dislocations
or persistent risk premia.
From a modeling perspective, this feature is conse-
quential. Linear specifications that treat volatility-
scaled exposure as approximately proportional to
changes in volatility may fail to capture the asymmet-
ric response of leverage across regimes. Flexible non-
linear architectures are better suited to accommodate
the threshold-like behavior induced by volatility tar-
geting, especially in settings where exposure amplifi-
cation during low-volatility periods magnifies the ef-
fect of predictive signals, while exposure compression
during high-volatility episodes dampens their impact.
The learning target is constructed as the volatility-
scaled next-period return,
```

```
targett= clip
```

######

```
rt+1
σt
```

###### ,− 20 , 20

######

###### , (23)

```
where clipping is applied to limit the influence of ex-
treme realizations during training.
```

```
Figure 6: Distribution of Momentum Features.
```

```
Figure 7: Induced Exposure by Volatility Targeting.
```

#### A.6 Stylized Facts of Returns and

#### Volatility

Daily returns exhibit strong deviations from Gaus-
sianity, with heavy tails and excess kurtosis that im-
ply substantial downside risk and state dependence.
Realized volatility is highly skewed and persistent, re-

```
flecting clustering and regime-dependent risk dynam-
ics. These features challenge linear predictive models
that rely on homoskedastic or symmetric error as-
sumptions.
```

#### A.7 Formal Statistical Diagnostics

```
To formally substantiate the distributional character-
istics suggested by the graphical evidence, we conduct
standard diagnostic tests.
First, normality of returns is rejected at conven-
tional significance levels using the Jarque–Bera test
across the majority of instruments, consistent with
the heavy tails observed in Figure 5. Excess kurto-
sis and skewness statistics further confirm substantial
deviations from Gaussianity.
Second, we test for non-stationarity in price levels
using augmented Dickey–Fuller tests. As expected
for financial price series, unit roots cannot generally
be rejected in levels, while returns exhibit stationar-
```

ity. This is consistent with the standard representa-
tion of asset prices as integrated processes with sta-
tionary increments.
Third, volatility persistence is assessed via auto-
correlation of squared returns and realized volatility.
We observe slow decay in the autocorrelation func-
tion, confirming volatility clustering and regime de-
pendence.
Taken together, these diagnostics confirm that the
dataset exhibits canonical stylized facts of financial
time series: heavy tails, conditional heteroskedastic-
ity, and non-stationarity in levels.

#### A.8 Implications for Modeling

Taken together, the distributional evidence high-
lights a fundamental feature of the data. While
the input predictors are largely bounded, symmet-
ric, and well-behaved, the target variable—realized
returns—exhibits extreme kurtosis, heavy tails, and
strong state dependence. This mismatch implies that
the primary modeling challenge lies not in stabilizing
the inputs, but in accurately capturing the condi-
tional distribution of returns given these inputs.
These empirical properties motivate the use of
flexible nonlinear architectures that can accommo-
date interactions, threshold effects, and regime-
dependent behavior. Nonlinear dependencies and
regime changes are frequently observed in financial
time series and are captured most effectively by non-
linear models [39]. Neural network architectures such
as LSTMs have shown superior performance in cap-
turing complex nonlinear patterns in volatility and
return dynamics [40]. In environments where small
changes in predictors can, under certain conditions,
lead to disproportionately large changes in outcomes,
models that adapt their functional form across the
state space are particularly well suited.

#### A.9 Scope and External Validity

While the dataset exhibits statistical properties com-
monly observed in financial markets, the conclusions
drawn in this study are, strictly speaking, conditional
on the data considered. As with any empirical in-
vestigation, it is not possible to establish universal

```
generality beyond the sampled instruments and time
period.
That said, the presence of heavy-tailed returns,
volatility clustering, and regime-dependent dynamics
suggests that the dataset captures structural charac-
teristics typical of many financial time series. For this
reason, we view the benchmark as representative of a
broad class of cross-asset forecasting problems. Nev-
ertheless, extending the analysis to alternative mar-
kets, frequencies, or macroeconomic environments re-
mains an important direction for future research.
```

## B Architecture Components

#### B.1 Linear Baselines

```
Autoregressive Model (AR1x). We consider an
AR(1) process
```

```
yt= φyt− 1 + εt, εt∼N (0,σ^2 ), (24)
```

```
which captures short-term autocorrelation. This
model assumes 1 one input feature. AR1x simply
applies AR(1) to each feature independently (for mul-
tiple input features).
```

```
DLinear and NLinear. DLinear and NLinear [15]
apply linear mappings to the input window:
```

```
ˆty = WXt+ b. (25)
```

```
DLinear decomposes the input into trend and sea-
sonal components, while NLinear operates on nor-
malized inputs. Both models lack temporal state and
serve as non-recurrent linear baselines.
```

#### B.2 Transformer Background

```
Transformer architectures [1] compute self-attention
quadratically as
```

```
Attn(Q,K,V ) = softmax
```

######

###### QK⊤

###### √

```
dk
```

######

###### V. (26)

```
Therefore, Transformer-based models have a large
number of trainable parameters.
```

#### B.3 State-Space Model Details

Mamba models [6, 29] implement selective state-
space models of the form

```
ht= Atht− 1 + Btxt, yt= Ctht, (27)
```

where htdenotes the latent state, xtthe input, and yt
the output. The matrices At, Bt, and Ctare initial-
ized using HiPPO LegS matrices [8], which provide
a principled discretization of continuous-time linear
dynamical systems.
In Mamba2, the state transition matrix is replaced
with a scaled identity, and the architecture incorpo-
rates a form of linear attention [7], improving effi-
ciency and numerical stability.
In this work, we employ a static HiPPO transi-
tion matrix and a fixed low-rank parameterization of
the step size ∆, resulting in a fixed temporal horizon.
This modification reduces sensitivity to noise and im-
proves robustness when modeling financial time se-
ries.

#### B.4 LSTM-based models

The LSTM architecture is included as a canonical
gated recurrent model that addresses the vanishing
gradient limitations of standard RNNs through an
additive memory cell and multiplicative gating. The
forget and input gates enable adaptive control over
memory retention and update, effectively implement-
ing a data-dependent filtering mechanism.
The LSTM [11] updates are given by

```
ft= σ(Wfxt+ Ufht− 1 + bf), (28)
it= σ(Wixt+ Uiht− 1 + bi), (29)
ot= σ(Woxt+ Uoht− 1 + bo), (30)
ct= ft⊙ ct− 1 + it⊙ tanh(Wcxt), (31)
ht= ot⊙ tanh(ct), (32)
```

where ft, it, and otdenote the forget, input, and out-
put gates, respectively. Such adaptive memory is par-
ticularly relevant in financial time series, where struc-
tural breaks and regime shifts render fixed-memory
models suboptimal. The gating mechanism allows
the model to dynamically adjust the effective time
horizon over which past information is retained.

```
Moreover, the nonlinear hidden-state representa-
tion provides a flexible mechanism for extracting pre-
dictive structure from low signal-to-noise and non-
Gaussian data, making LSTMs a natural and widely
adopted baseline in financial forecasting tasks.
```

```
xLSTM. We provide a concise technical descrip-
tion of xLSTM following [12] (Eq. 8–17).
```

```
Exponential Gating. In contrast to classical
LSTMs, which employ sigmoid gates ft= σ(·) and
it= σ(·), xLSTM replaces sigmoid activations with
exponential gating followed by normalization. Let
```

```
f ̃t= Wfxt+ Rfht− 1 + bf, (33)
̃it= Wixt+ Riht− 1 + bi, (34)
```

```
which are the pre-activation gates, similar to
LSTM. The raw gates are exponentiated,
```

```
fˆt= exp
```

######

```
f ̃t
```

######

```
, ˆit= exp
```

###### ̃

```
it
```

######

###### , (35)

```
and subsequently normalized to ensure numerical
stability and controlled memory growth. In log-
domain form, this normalization is implemented via
a running maximum term to prevent overflow (see
[12], Eq. 15–17). The resulting gates satisfy a convex
combination structure analogous to classical LSTM
gating but without sigmoid saturation.
```

```
sLSTM (Scalar LSTM). The scalar variant
maintains a single memory state per unit. The up-
date equations are
```

```
̃tc = tanh(Wzxt+ Rzht− 1 + bz), (36)
ct= ft⊙ ct− 1 + it⊙ ̃tc, (37)
nt= ft⊙ nt− 1 + it, (38)
ht= ot⊙
ct
nt
```

###### , (39)

```
where ft, it, and otdenote normalized exponential
forget, input, and output gates, respectively, and nt
is a stabilizing normalizer. The division by ntensures
scale control of the memory state.
```

mLSTM (Matrix LSTM). The matrix variant
generalizes the memory to a matrix-valued state Ct∈
Rd×d. At each step, key–value vectors kt,vt∈ Rd
are generated and stored via a gated outer-product
update:

```
Ct= ft⊙ Ct− 1 + it⊙ (vtkt⊤). (40)
```

The hidden state is retrieved through a query vec-
tor qt,

```
ht= Ctqt, (41)
```

This Bidirectional Associative Memory (BAM)
[41–44] setting yields high separability between
stored patterns and allows efficient recall of past in-
formation. Importantly, mLSTM removes state com-
pression recurrence but still has temporal dependency
in memory accumulation. It is more parallelizable,
but not fully parallel.

Patch sLSTM (PsLSTM). Given a multivari-
ate time series X ∈ RL×d, each channel is treated
independently and segmented into non-overlapping
patches of length ℓ:

```
̃x(pi)= Patch
```

######

```
x((ip)−1)ℓ+1:pℓ
```

######

```
, i = 1,...,d. (42)
```

Each patch embedding is processed by an sLSTM
with shared parameters across channels:

```
h(pi)= sLSTM(h(pi−) 1 , ̃x(pi)). (43)
```

Parameter sharing preserves channel independence
while reducing model complexity and mitigating
overfitting.

#### B.5 Hybrid Architecture Details

The hybrids are designed to improve robustness in
noisy financial time series by enhancing the signal-to-
noise ratio, enabling adaptive feature selection, and
stabilizing temporal state updates.

```
Variable Selection Network (VSN) The Vari-
able Selection Network (VSN), inspired by the Tem-
poral Fusion Transformer [5], performs feature-wise
nonlinear embedding followed by dynamic soft selec-
tion of relevant covariates at each time step.
Given an input vector xt ∈ RC consisting of C
covariates, each variable is embedded independently:
```

```
ht,i= φi(xt,i), i = 1,...,C, (44)
```

```
where φi(·) denotes a learnable nonlinear embedding
function.
The embeddings are concatenated and passed
through a gating network to compute feature impor-
tance weights:
```

```
αt= softmax (Wg[ht, 1 ,...,ht,C] + bg), (45)
```

```
where Wgand bgare learnable parameters.
The selected representation is computed as a
weighted sum:
```

```
̃xt=
```

###### XC

```
i=1
```

```
αt,iht,i. (46)
```

```
This mechanism enables adaptive suppression of
noisy or uninformative covariates and improves ro-
bustness in non-stationary environments.
```

```
VSN+LSTM (VLSTM) The VSN+LSTM (VL-
STM) model combines VSN-based feature selection
with recurrent temporal encoding. At each time step,
the input vector is processed by a VSN to produce a
dynamically weighted feature representation:
```

```
̃xt= VSN(xt). (47)
```

```
The resulting sequence { ̃xt}Lt=1 is then passed
through an LSTM to construct a compact temporal
state:
(ht,ct) = LSTM( ̃xt,ht− 1 ,ct− 1 ). (48)
```

```
For one-step-ahead forecasting, the prediction is
obtained from the final hidden state:
```

```
ˆLy +1= WohL+ bo. (49)
```

VSN–Mamba2 In the VSN–Mamba2 hybrid, fea-
ture selection and temporal modeling are explicitly
decoupled. Given an input xt∈ Rd, the VSN com-
putes feature-wise importance weights:

```
αt= softmax(g(xt)), ̃xt= αt⊙ xt, (50)
```

where g(·) denotes a learnable gating network and ⊙
denotes element-wise multiplication.
The filtered input ̃xtis then passed to the Mamba2
state-space block:

```
ht= Aht− 1 + B ̃xt, (51)
```

where A and B denote the state transition and in-
put matrices, respectively. This design improves ro-
bustness by reducing the influence of noisy covariates
prior to temporal state updates.

LSTM + PatchTST This hybrid architecture
combines explicit recurrence with attention by using
an LSTM as a channel-wise temporal denoiser prior
to PatchTST. Each input channel is processed inde-
pendently using a shared LSTM backbone:

```
h(ti)= LSTM(x(ti),h(ti−) 1 ), i = 1,...,d, (52)
```

where x(ti)denotes the i-th feature.
The resulting hidden states are segmented into
temporal patches and passed to PatchTST, which ap-
plies self-attention over patches:

```
̃hp= PatchTST({h(ti)}t∈Pp), (53)
```

where Ppdenotes the set of time steps belonging to
patch p.
This separation of concerns allows the LSTM to
stabilize local temporal structure while PatchTST ag-
gregates medium- and long-range dependencies.

VSN + xLSTM In the VSN + xLSTM hybrid,
the sequence of VSN-selected representations is di-
rectly fed into an xLSTM. The model maintains a
matrix-valued memory state Mt∈ RH×H, which is
updated recursively:

```
(Mt,st) = xLSTM( ̃xt,Mt− 1 ), (54)
```

```
where st ∈ RH denotes the output state. The
matrix-valued memory enables modeling of higher-
order temporal interactions and long-range depen-
dencies beyond the capacity of vector-based recurrent
architectures.
```

#### B.6 Temporal Fusion Transformer

#### Details

```
TFT combines recurrent encoding with attention-
based aggregation. Given an input sequence{xt}Lt=1,
a recurrent encoder produces latent states
```

```
ht= LSTM(xt,ht− 1 ).
```

```
Multi-head attention is then applied to aggregate in-
formation across time:
```

```
ˆty = Attn(h1:t).
```

```
Variable selection networks, gating layers, and static
covariate encoders further modulate the representa-
tions to improve interpretability and robustness.
```

## C Model Configurations and

## Hyperparameter Selection

```
To ensure a fair and systematic comparison across
architectures, we define structured hyperparameter
search spaces tailored to each model family. All
models are evaluated under comparable training bud-
gets, with architecture-specific parameters varied
only where structurally relevant. Tables 6 and 7 sum-
marizes the general configuration ranges.
```

```
General Training Configuration. For each
model, we tune the batch size, hidden dimen-
sionality (H), learning rate, and input sequence
length. Learning rates are selected from logarith-
mic ranges between 10 −^4 and 10 −^2 or from discrete
sets { 10 −^3 , 5 × 10 −^4 , 10 −^4 } depending on architec-
tural stability. Sequence lengths are chosen to re-
flect short- and medium-horizon dependencies (e.g.,
64–512 timesteps), while maintaining comparability
across models.
```

Architecture-Specific Hyperparameters. In
addition to shared training parameters, each model
family includes structural hyperparameters:

- Transformer-based models (e.g., TFT,
  PatchTST) vary the number of attention heads,
  encoder layers, and sparsity factors.
- State-space models (Mamba2 variants) tune
  the number of layers, convolution kernel size,
  SSM expansion factors, rank, and discretization
  parameters (e.g., ∆max, HiPPO scaling).
- xLSTM-based models vary the number of
  stacked blocks, convolutional kernel sizes, and
  projection expansion factors.
- Patch-based models additionally tune patch
  length and stride fraction.
- Classical baselines (AR1, DLinear) are evalu-
  ated with minimal architectural tuning.

Search Protocol. Hyperparameter selection is
performed via grid search over the predefined discrete
ranges. Each configuration is trained independently,
and the best-performing setting is selected based on
validation Sharpe Ratio. Importantly, the same val-
idation procedure and performance metric are used
for all architectures to avoid selection bias.

Training Details. We employ the ADAM opti-
mizer [45] for all models. Early stopping is applied
with a patience of 20 epochs. For validation, the last
10% of the training data is reserved as a validation
set. Gradient clipping based on the gradient norm is
used to stabilize training.
Models are retrained every five years using a
rolling-window scheme. For strategies requiring an
initial lookback period, performance metrics such as
the Sharpe ratio are not evaluated during the burn-in
phase. In particular, for models with a fixed initial-
ization window we use L 0 = 21, while for sequence
models the burn-in period corresponds to one quar-
ter of the input sequence length (e.g., 21 for sequence
length 84 and 128 for sequence length 512).

```
Model Capacity Considerations. The number
of trainable parameters varies substantially across
models (Table 6), reflecting inherent architectural
differences. We do not explicitly match parameter
counts, as doing so would distort the native design of
certain architectures (e.g., state-space vs. attention-
based models). Instead, we control for training pro-
tocol, optimization strategy, and evaluation metric to
ensure comparability.
This structured configuration framework ensures
that performance differences arise from architectural
properties rather than inconsistent tuning practices.
```

## D Performance Metrics and

## Evaluation Criteria

```
This appendix describes the performance, risk, and
robustness measures used throughout the empirical
evaluation. Given the low signal-to-noise ratio and
heavy-tailed nature of financial returns, we rely on
a broad set of complementary metrics to assess not
only average performance, but also statistical signif-
icance, downside risk, trading intensity, and incre-
mental value relative to a passive benchmark.
```

#### D.1 Return and Risk-Adjusted Per-

#### formance

```
Annualized Return. Annualized return is com-
puted as the mean daily portfolio return scaled by
the number of trading days per year. While intuitive,
this metric does not account for risk and is therefore
interpreted jointly with risk-adjusted measures.
```

```
Compound Annual Growth Rate (CAGR).
CAGR measures the geometric average annual
growth of portfolio value over the evaluation period:
```

###### CAGR =

######

###### VT

###### V 0

###### 1 /T

###### − 1 ,

```
where V 0 and VTdenote initial and terminal portfo-
lio values and T is the length of the sample in years.
CAGR reflects long-run capital accumulation and pe-
nalizes volatility drag.
```

```
Table 6: Model configurations and general hyperparameter search ranges.
```

```
Model Trainable Params Batch Size Hidden Dim (H) LR Range Seq Len Dropout
Mamba2+VSN 3,571,348 {32,64,128} {32,64,128,256} 10 −^4 – 10 −^284 {0.1,0.2,0.3,0.4}
xLSTM 2,507,269 {128,256} {64,128,256} {1e-3,5e-4,1e-4} 84 {0.2,0.3,0.4,0.5}
xLSTM+VSN 6,146,368 {128,256} {64,128,256} {1e-3,5e-4,1e-4} 84 {0.2,0.3,0.4,0.5}
PsLSTM 1,963,841 {16,32} {16,32,64} {1e-3,5e-4,1e-4} 512 {0.2,0.3,0.4,0.5}
VLSTM 1,142,963 {128,256} {64,128,256} 10 −^4 – 10 −^284 {0.2,0.3,0.4,0.5}
LSTM 73,729 {128,256} {64,128,256} 10 −^4 – 10 −^284 {0.2,0.3,0.4,0.5}
PatchTST 1,139,739 {16,32} {16,32,64,96} {1e-3,5e-4} 512 {0.1,0.2,0.3,0.4,0.5}
LSTM+PatchTST 634,881 {16,32} {16,32,64,96} {1e-3,5e-4} 512 {0.1,0.2,0.3,0.4,0.5}
Mamba2 18,882 {32,64,128} {32,64,128,256} 10 −^4 – 10 −^284 {0.1,0.2,0.3,0.4}
DLinear 2,111 {256,512,1024} {64,128,256} 10 −^4 – 10 −^264 {0.1,0.2,0.3}
AR1 2,073 {128,256} {64,128,256} 10 −^4 – 10 −^284 {0.2,0.3,0.4,0.5}
TFT 347,507 {128,256} {64,128,256} 10 −^4 – 10 −^284 {0.2,0.3,0.4,0.5}
```

```
Table 7: Architecture-specific hyperparameter search spaces.
```

#### Architecture Hyperparameter Search Space

#### Transformer-based Heads: {1,2,4}; Layers: {3,4,5,6}; Sparsity: {2,3,4,5,6}

#### Mamba / SSM Layers: {1,2,3,5}; Kernel: {2,3,5,7,9}; Conv: {2,8,16,32}

#### Rank: {4,8,16,32}; ∆max: {0.2,0.4,0.6,0.8}; HiPPO: 0.1–0.5

#### xLSTM Blocks: {1–6}; Kernel: {1,2,4,6,7,9}; Projection: {1–2.5}

#### Patch-based Patch Length: {4,8,16,32,64}; Stride: {0.25,0.5,1}

#### Classical (AR1, DLinear) Layers: {2,3,5}

Sharpe Ratio. The Sharpe ratio is defined as the
mean excess return divided by the standard deviation
of returns. All reported Sharpe ratios are annualized.
This is the primary optimization objective and head-
line performance metric throughout the study.

Information Ratio. The Information Ratio mea-
sures risk-adjusted excess performance relative to a
passive buy-and-hold benchmark:

###### IR =

```
Eˆ[rt− rpassivet ]
q
Varˆ(rt− rpassivet )
```

###### .

This metric captures incremental value beyond mar-
ket exposure.

#### D.2 Statistical Significance

```
HAC-Adjusted t-Statistics. To assess statisti-
cal significance under serial correlation and het-
eroskedasticity, we report Newey–West heteroskedas-
ticity and autocorrelation consistent (HAC) t-
statistics for mean returns and Sharpe ratios. This
adjustment is critical in financial time series, where
returns often exhibit time dependence.
```

```
HAC t-Statistic versus Passive. We addition-
ally report HAC-adjusted t-statistics for excess re-
turns relative to the passive benchmark, testing
whether observed outperformance is statistically dis-
tinguishable from zero after accounting for depen-
dence in relative returns.
```

#### D.3 Directional Accuracy and Trad-

#### ing Activity

Hit Rate. The hit rate measures the fraction of pe-
riods in which the strategy correctly predicts the sign
of returns. While not sufficient for profitability on its
own, it provides insight into directional consistency.

Turnover. Turnover is defined as the average ab-
solute change in portfolio weights across consecutive
periods. High turnover implies greater transaction
costs and reduced implementability.

Turnover (xGMV). Turnover expressed as a mul-
tiple of gross market value (xGMV) provides a scale-
free measure of trading intensity and facilitates com-
parison across strategies.

#### D.4 Downside Risk and Tail Behavior

Maximum Drawdown. Maximum drawdown is
the largest peak-to-trough decline in cumulative port-
folio value. It captures worst-case capital loss and is
a key risk metric for real-world deployment.

Calmar Ratio. The Calmar ratio is defined as
CAGR divided by maximum drawdown. It measures
return efficiency relative to extreme downside risk
and complements the Sharpe ratio.

Conditional Value-at-Risk (CVaR 5%). CVaR
at the 5% level measures the expected loss conditional
on returns falling in the worst 5% of outcomes. Un-
like Value-at-Risk, CVaR captures tail severity and is
particularly relevant in heavy-tailed financial return
distributions.

#### D.5 Benchmark-Relative Diagnostics

Correlation versus Passive. We report the Pear-
son correlation between strategy returns and the pas-
sive benchmark. Lower correlation indicates greater
diversification benefits and reduced dependence on
market direction.

```
Profit and Loss (PnL). Cumulative profit and
loss (PnL) curves are used for visual comparison of
strategies over time. PnL trajectories provide in-
sight into path dependence, drawdown behavior, and
regime sensitivity that may not be evident from sum-
mary statistics alone.
Taken together, these metrics provide a compre-
hensive and economically meaningful assessment of
model performance, balancing average returns, sta-
tistical reliability, downside risk, trading realism, and
incremental value over a passive investment strategy.
```

## E Asset-level Results

```
This section provides a detailed breakdown of model
performance at the individual asset level. The per-
asset analysis presented here serves two purposes: (i)
to evaluate cross-sectional robustness and (ii) to as-
sess whether performance is concentrated in specific
asset classes or broadly distributed across markets.
We use breakeven transaction cost, which is defined
as the constant cost per unit of leveraged turnover
that drives total gross PnL to zero. Formally,
```

```
c∗=
```

###### P

```
tR
```

```
gross
Pt
tτt
```

###### ,

```
where τtdenotes leveraged turnover. If actual trad-
ing costs remain below c∗, the strategy remains prof-
itable; the ratio c∗/cactualmeasures implementation
robustness.
Results are grouped by asset category (Bond, Com-
modity, Energy, FX, and Index futures). For each
model, we report cumulative PnL trajectories per
asset as well as box-and-whisker summaries of the
Sharpe Ratio distributions. This allows us to ex-
amine both temporal consistency and cross-sectional
dispersion.
```

#### E.1 Asset Description

```
Table 8 reports the full universe of tradable assets
included in the empirical analysis. The cross-asset
dataset spans foreign exchange, equity indices, fixed
income, energy, and agricultural and metal com-
modities, including both pit and electronic contracts
```

```
Table 8: Asset universe and classification.
```

```
Ticker Group Description Ticker Group Description Ticker Group Description
AN FX Australian dollar comp BC Energy Brent crude oil BG Energy Brent gasoil
BN FX British pound comp CA Index CAC 40 index CADJPY FX CAD JPY cross
CB Bond Canada 10Y bond CC Comdty Cocoa CN FX Canadian dollar comp
CR Comdty CRB index CT Comdty Cotton No.2 DA Comdty Milk III
DT Bond Euro Bund DX FX US dollar index EN Index Nasdaq mini
ER Index Russell 2000 mini ES Index S&P 500 mini FB Bond US 5Y note comp
FN Index Euro Stoxx comp GI Comdty Goldman Sachs idx GS Bond UK Gilt long
HS Index Hang Seng JN FX Japanese yen comp JO Comdty Orange juice
KC Comdty Coffee KW Comdty Wheat KC LB Comdty Lumber
LX Index FTSE 100 MD Index S&P 400 mini MP FX Mexican peso
MW Comdty Wheat Minneapolis NK Index Nikkei 225 NOKUSD FX NOK USD cross
NR Comdty Rough rice SB Comdty Sugar No.11 SN FX Swiss franc comp
TU Bond US 2Y note comp TY Bond US 10Y note comp UB Bond Euro Bobl
US Bond US T-bond comp USDNZD FX USD NZD cross USDSEK FX USD SEK cross
USDSGD FX USD SGD cross UZ Bond Euro Schatz XU Index Euro Stoxx 50
XX Index STOXX 50 YM Index Mini Dow ZA Comdty Palladium elec
ZB Energy RBOB elec ZC Comdty Corn elec ZF Comdty Feeder cattle elec
ZG Comdty Gold elec ZI Comdty Silver elec ZK Comdty Copper elec
ZL Comdty Soybean oil elec ZM Comdty Soybean meal elec ZN Energy Natural gas elec
ZO Comdty Oats elec ZP Comdty Platinum elec ZR Comdty Rough rice elec
ZS Comdty Soybeans elec ZT Comdty Live cattle elec ZU Energy Crude oil elec
ZW Comdty Wheat elec ZZ Comdty Lean hogs elec
```

where available. This broad coverage ensures sub-
stantial cross-sectional and cross-asset heterogeneity,
enabling the evaluation of model robustness across di-
verse liquidity conditions, macroeconomic exposures,
and volatility regimes.

#### E.2 VSLTM

Figures 8–12 display cumulative PnL per asset across
the five asset categories. The corresponding box-and-
whisker plots (Figures 13–17) summarize the distri-
bution of Sharpe Ratio for each instrument.
Several observations emerge. First, performance is
not driven by a single dominant contract but is dis-
tributed across multiple assets within each category.
Second, tail behavior, as captured in the box plots,
suggests that risk-adjusted performance is primarily
driven by stable median returns rather than isolated
extreme gains.
Overall, the per-asset analysis indicates that VL-
STM’s aggregate performance is supported by con-
sistent cross-sectional contributions rather than con-
centration in a small subset of instruments.

```
Figure 8: VLSTM PnL per asset - bond Futures
```

Figure 9: VLSTM PnL per asset - Commodities Futures

```
Figure 10: VLSTM PnL per asset - Index Futures
```

Figure 11: VLSTM PnL per asset - Energy Futures

Figure 12: VLSTM PnL per asset - FX Futures

Figure 13: VLSTM box and whisker annual Sharpe Ratio
per asset - Bond Futures

Figure 14: VLSTM box and whisker annual Sharpe Ratio
per asset - Commodities Futures

```
Figure 15: VLSTM box and whisker annual Sharpe Ratio
per asset - Index Futures
```

```
Figure 16: VLSTM box and whisker annual Sharpe Ratio
per asset - Energy Futures
```

Figure 17: VLSTM box and whisker annual Sharpe Ratio
per asset - FX Futures

```
Table 9: VLSTM: Annualised Volatility-Rescaled Performance and Breakeven Transaction Costs (bps)
```

```
Ticker Gross (ann.) Turnover (ann.) c∗(bps) Ticker Gross (ann.) Turnover (ann.) c∗(bps)
LB 0.2485 80.58 30.84 ZO 0.2027 84.01 24.13
DA 0.2529 114.78 22.04 ZN 0.0618 45.04 13.73
JO 0.0986 74.19 13.29 EN 0.1003 86.69 11.57
KC 0.0738 63.92 11.54 ZR 0.1339 137.44 9.75
BC 0.0512 61.26 8.36 ZF 0.1279 157.12 8.14
NR 0.1086 137.78 7.88 NK 0.0706 89.77 7.87
ES 0.0796 107.12 7.43 KW 0.0686 93.77 7.32
ZL 0.0827 115.79 7.15 ZA 0.0558 82.87 6.74
ZI 0.0444 68.93 6.44 ZU 0.0381 64.42 5.91
ZS 0.0476 83.69 5.68 MW 0.0669 127.30 5.25
XU 0.0507 98.17 5.17 XX 0.0614 118.76 5.17
ZK 0.0421 83.07 5.06 CT 0.0421 91.73 4.59
ZW 0.0417 91.08 4.57 SB 0.0313 80.33 3.89
MD 0.0442 115.53 3.82 HS 0.0301 85.51 3.52
ZC 0.0363 103.71 3.50 ZT 0.0464 135.72 3.42
ER 0.0264 80.13 3.30 CA 0.0267 92.21 2.90
CR 0.0389 138.25 2.82 BG 0.0216 78.91 2.73
DT 0.0782 291.86 2.68 MP 0.0368 143.21 2.57
ZM 0.0215 83.55 2.57 GI 0.0230 94.01 2.44
ZB 0.0160 66.47 2.41 NOKUSD 0.0382 165.09 2.31
USDSEK 0.0419 188.00 2.23 GS 0.0547 282.39 1.94
JN 0.0321 209.92 1.53 ZG 0.0177 119.59 1.48
CC 0.0115 78.16 1.47 USDNZD 0.0209 169.47 1.23
USDSGD 0.0422 368.09 1.15 UB 0.0664 591.04 1.12
ZP 0.0110 101.90 1.08 TY 0.0415 411.13 1.01
DX 0.0284 298.04 0.95 LX 0.0101 109.94 0.91
YM 0.0085 106.28 0.80 CN 0.0199 272.83 0.73
CB 0.0194 344.26 0.56 FN 0.0130 233.13 0.56
CADJPY 0.0080 151.64 0.53 US 0.0104 198.09 0.52
FB 0.0281 689.19 0.41 UZ 0.0508 1940.20 0.26
TU 0.0094 2218.65 0.04 ZZ -0.0004 80.57 -0.05
BN -0.0092 207.76 -0.44 AN -0.0170 164.71 -1.03
SN -0.0272 212.34 -1.28
```

#### E.3 xLSTM

Figures 18–22 present cumulative PnL per asset for
xLSTM across the five asset groups. The associated
box-and-whisker plots (Figures 23–27) provide distri-
butional summaries.
Relative to VLSTM, xLSTM exhibits stronger
cross-sectional homogeneity in certain categories. In
higher-volatility sectors such as energy and commodi-
ties, dispersion increases, but median performance
remains positive across most instruments. This sug-
gests that the model adapts to heterogeneous volatil-
ity structures without excessive tail risk concentra-
tion.
Importantly, no systematic degradation is observed
in a specific asset class, indicating that performance
is not regime- or sector-dependent. The cross-asset
consistency observed in the box plots further sup-
ports the robustness of the learned representations.

```
Figure 18: xLSTM PnL per asset - bond Futures
```

Figure 19: xLSTM PnL per asset - Commodities Futures

```
Figure 20: xLSTM PnL per asset - Index Futures
```

Figure 21: xLSTM PnL per asset - Energy Futures

Figure 22: xLSTM PnL per asset - FX Futures

Figure 23: xLSTM box and whisker annual Sharpe Ratio
per asset - Bond Futures

Figure 24: xLSTM box and whisker annual Sharpe Ratio
per asset - Commodities Futures

```
Figure 25: xLSTM box and whisker annual Sharpe Ratio
per asset - Index Futures
```

```
Figure 26: xLSTM box and whisker annual Sharpe Ratio
per asset - Energy Futures
```

Figure 27: xLSTM box and whisker annual Sharpe Ratio
per asset - FX Futures

```
Table 10: xLSTM: Annualised Volatility-Rescaled Performance and Breakeven Transaction Costs (bps)
```

```
Ticker Gross (ann.) Turnover (ann.) c∗(bps) Ticker Gross (ann.) Turnover (ann.) c∗(bps)
LB 0.2338 68.93 33.92 DA 0.2200 83.29 26.41
ZO 0.2157 98.88 21.81 ES 0.0501 26.01 19.25
ZN 0.0544 29.02 18.76 CC 0.0419 24.23 17.31
EN 0.0572 38.19 14.97 JO 0.1375 96.18 14.30
ZR 0.1818 132.86 13.68 NR 0.1489 135.22 11.01
LX 0.0430 40.14 10.72 BG 0.0443 51.13 8.66
ZM 0.0279 32.54 8.57 KW 0.0587 75.50 7.77
YM 0.0409 55.80 7.33 MW 0.0824 116.51 7.07
ZT 0.0919 131.96 6.96 MP 0.0419 61.25 6.83
DT 0.0501 79.15 6.33 ZA 0.0437 76.56 5.71
ZW 0.0415 84.22 4.93 ZC 0.0471 101.33 4.65
MD 0.0325 71.83 4.53 JN 0.0336 74.70 4.50
ZZ 0.0167 42.08 3.96 ZF 0.0580 161.56 3.59
ZI 0.0211 60.24 3.50 ZS 0.0121 38.73 3.12
ER 0.0163 52.64 3.09 USDSEK 0.0286 100.11 2.85
ZL 0.0314 118.88 2.64 ZB 0.0107 41.06 2.62
XU 0.0385 151.47 2.54 UB 0.0305 176.77 1.72
XX 0.0192 178.70 1.07 CB 0.0177 217.61 0.81
GS 0.0133 223.73 0.60 NK 0.0055 145.81 0.38
ZU 0.0011 38.32 0.29 ZK 0.0032 115.80 0.28
UZ 0.0145 584.56 0.25 FB 0.0087 441.46 0.20
TU 0.0205 1121.50 0.18 TY 0.0034 219.72 0.16
USDSGD 0.0010 208.81 0.05 NOKUSD -0.0028 102.23 -0.27
CN -0.0106 193.52 -0.55 US -0.0086 101.15 -0.85
CADJPY -0.0144 166.63 -0.86 GI -0.0072 54.74 -1.31
DX -0.0359 270.55 -1.33 BN -0.0143 95.74 -1.49
CA -0.0167 110.31 -1.52 SB -0.0084 41.65 -2.01
HS -0.0103 45.01 -2.30 FN -0.0193 77.30 -2.50
AN -0.0371 131.94 -2.81 ZP -0.0301 100.81 -2.99
ZG -0.0190 63.04 -3.02 USDNZD -0.0386 92.40 -4.17
CT -0.0397 92.56 -4.29 KC -0.0218 43.61 -4.99
SN -0.0737 125.71 -5.86 BC -0.0126 19.67 -6.40
```

Cross-Sectional Robustness. Taken together,
the complementary results reinforce the conclusions
drawn in the main text. Aggregate Sharpe ratios are
not driven by isolated outliers but reflect broadly dis-
tributed performance across asset classes. Differences
between models are manifested not only in overall
portfolio metrics but also in dispersion characteris-
tics and category-level stability.
This granular evaluation is particularly relevant in
financial forecasting, where apparent portfolio-level
improvements may otherwise mask instability or con-
centration risk at the instrument level.

## F Annual Sharpe Ratio

Table 11 reports annual out-of-sample Sharpe ratios
for all benchmark strategies over the period 2010–

2024. The year-by-year decomposition complements
      the aggregated results in the main text by providing

```
a finer assessment of temporal stability and regime
dependence. Several patterns emerge. First, perfor-
mance is not concentrated in a single favorable subpe-
riod: the leading deep sequence models exhibit con-
sistently positive Sharpe ratios across a broad range
of market environments, including the post-crisis re-
covery, the low-volatility mid-2010s expansion, and
the high-uncertainty period following 2020. Sec-
ond, while cross-sectional performance rankings vary
from year to year—as expected given structural shifts
in volatility and cross-asset correlations—the top-
performing architectures remain competitive in most
years and avoid persistent underperformance. Third,
classical linear benchmarks display greater sensitivity
to adverse regimes, with more frequent negative an-
nual Sharpe ratios. Overall, the annual breakdown
confirms that the superior aggregated performance
documented in the main text is not driven by isolated
episodes, but rather reflects sustained risk-adjusted
returns across heterogeneous market conditions.
```

```
Table 11: Annual Sharpe Ratios by Strategy (2010–2024)
```

Strategy 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024
AR1x 1.45 -0.07 -0.11 0.23 2.97 -0.02 0.05 0.67 -0.59 0.18 3.02 0.63 1.37 1.45 0.30
ARnx 1.60 0.06 -0.25 0.06 2.51 0.22 -0.17 0.30 -0.90 0.48 2.07 0.90 1.64 1.11 -0.16
DLinear 1.21 -0.11 -0.43 -0.29 2.89 0.33 -0.19 0.42 -0.28 -0.27 3.22 0.35 1.45 2.30 -0.94
LSTM 3.58 0.29 2.47 0.74 1.81 2.09 2.19 -1.19 1.72 3.20 -1.51 1.12 3.44 1.22 1.06
VLSTM 3.96 1.48 3.92 1.13 3.03 1.90 4.03 1.06 1.33 4.74 0.10 2.81 3.82 2.76 -0.10
Mamba2 0.88 -0.38 -0.71 0.07 3.22 0.18 0.15 0.68 -0.05 0.00 3.15 1.33 2.27 1.43 -0.49
VSN+Mamba2 1.48 0.27 -0.12 -0.00 3.43 0.65 0.54 1.31 -0.05 0.23 3.31 1.92 2.64 1.45 -0.63
PatchTST 2.69 0.48 0.61 -1.21 0.72 0.24 0.13 -0.13 1.14 1.49 1.20 1.98 0.85 0.54 0.59
LPatchTST 3.90 1.50 3.04 0.51 3.40 1.60 1.72 2.39 1.55 3.26 1.13 3.46 3.24 2.57 1.31
PsLSTM 2.71 1.38 1.92 -0.40 2.89 1.09 1.92 1.87 1.39 2.92 1.69 3.04 1.74 2.29 -0.35
TFT 3.97 1.21 3.90 1.19 3.04 1.50 3.18 1.05 1.32 3.36 -0.14 2.62 4.28 3.25 0.37
VxLSTM 2.73 1.34 1.71 0.22 3.25 0.11 1.92 2.82 0.60 1.95 3.77 2.24 2.13 1.88 -1.31
xLSTM 2.64 1.41 1.72 -0.28 2.91 0.34 2.29 2.96 0.46 2.38 3.83 2.62 1.55 1.96 0.02
iTransformer 1.30 0.20 -1.14 0.19 2.34 0.70 -0.07 0.65 -1.16 0.21 1.43 0.95 -1.03 0.68 0.46
Mamba 0.82 -0.27 -0.46 -0.48 3.03 0.41 0.07 0.04 -0.17 -0.40 1.29 1.00 -0.54 1.47 -0.40
NLinear 1.18 0.38 -1.20 0.18 2.60 0.44 -0.22 0.96 -1.12 0.63 3.02 0.60 0.09 1.35 1.07
