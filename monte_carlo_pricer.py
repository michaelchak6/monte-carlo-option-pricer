"""
Monte Carlo Option Pricer
=========================
Prices European call and put options using Monte Carlo simulation
with Geometric Brownian Motion (GBM) path generation.

Features:
- European call/put pricing via Monte Carlo simulation
- Analytical Black-Scholes benchmark for validation
- Convergence analysis across simulation counts
- Sensitivity visualisations (spot price, volatility, time to expiry)
- 95% confidence intervals on MC estimates

Author: Michael Chak
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from scipy.optimize import brentq

# ── Styling ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "lines.linewidth":  1.8,
    "font.family":      "monospace",
})

BLUE    = "#58a6ff"
GREEN   = "#3fb950"
ORANGE  = "#f78166"
PURPLE  = "#bc8cff"
YELLOW  = "#e3b341"


# ── Black-Scholes Analytical Pricer ───────────────────────────────────────────
def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Analytical Black-Scholes price for a European option.

    Parameters
    ----------
    S           : float  – current spot price
    K           : float  – strike price
    T           : float  – time to expiry (years)
    r           : float  – risk-free rate (annual)
    sigma       : float  – volatility (annual)
    option_type : str    – 'call' or 'put'

    Returns
    -------
    price : float
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, option_type="call"):
    """Returns a dict of all five Greeks via Black-Scholes."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega  = S * pdf_d1 * np.sqrt(T) / 100          # per 1% move in vol
    theta_call = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                  - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    theta = theta_call if option_type == "call" else (
        theta_call + r * K * np.exp(-r * T) / 365)
    rho   = (K * T * np.exp(-r * T) * norm.cdf(d2)  if option_type == "call"
             else -K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100

    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho": rho}


# ── Implied Volatility via Brent's Method ─────────────────────────────────────
def implied_volatility(market_price, S, K, T, r, option_type="call"):
    """Solves for implied volatility using Brent's root-finding method."""
    objective = lambda sigma: black_scholes(S, K, T, r, sigma, option_type) - market_price
    try:
        return brentq(objective, 1e-6, 10.0, xtol=1e-8)
    except ValueError:
        return np.nan


# ── Monte Carlo Pricer ─────────────────────────────────────────────────────────
def monte_carlo_price(S, K, T, r, sigma, option_type="call",
                      n_simulations=100_000, seed=42):
    """
    Price a European option via Monte Carlo simulation using GBM.

    Returns
    -------
    price : float       – MC estimated option price
    ci    : (float, float) – 95% confidence interval (lower, upper)
    paths : np.ndarray  – simulated terminal stock prices
    """
    rng = np.random.default_rng(seed)
    Z   = rng.standard_normal(n_simulations)

    # GBM terminal stock price
    S_T = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

    # Payoffs
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)

    discounted = np.exp(-r * T) * payoffs
    price      = discounted.mean()
    std_err    = discounted.std() / np.sqrt(n_simulations)
    ci         = (price - 1.96 * std_err, price + 1.96 * std_err)

    return price, ci, S_T


# ── Convergence Analysis ───────────────────────────────────────────────────────
def convergence_analysis(S, K, T, r, sigma, option_type="call",
                         max_sims=100_000):
    """
    Compute MC price at increasing simulation counts to show convergence
    toward the analytical Black-Scholes price.
    """
    sim_counts = np.unique(np.logspace(1, np.log10(max_sims), 60).astype(int))
    mc_prices  = []
    ci_lower   = []
    ci_upper   = []

    for n in sim_counts:
        p, ci, _ = monte_carlo_price(S, K, T, r, sigma, option_type,
                                     n_simulations=n, seed=0)
        mc_prices.append(p)
        ci_lower.append(ci[0])
        ci_upper.append(ci[1])

    return sim_counts, np.array(mc_prices), np.array(ci_lower), np.array(ci_upper)


# ── Visualisation ─────────────────────────────────────────────────────────────
def plot_results(S=100, K=100, T=1.0, r=0.05, sigma=0.2,
                 option_type="call", n_simulations=100_000):
    """
    Generate an 8-panel figure covering:
      1. Simulated GBM terminal price distribution
      2. Convergence of MC price to BS analytical
      3. Call price vs spot price (MC vs BS)
      4. Put price vs spot price (MC vs BS)
      5. Option price vs volatility
      6. Option price vs time to expiry
      7. Greeks vs spot price
      8. Implied volatility smile
    """

    mc_price, ci, S_T = monte_carlo_price(S, K, T, r, sigma, option_type,
                                          n_simulations, seed=42)
    bs_price = black_scholes(S, K, T, r, sigma, option_type)

    fig = plt.figure(figsize=(20, 22))
    fig.suptitle("Monte Carlo Option Pricer  ·  European Options",
                 fontsize=16, fontweight="bold", color="#e6edf3", y=0.98)
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

    # ── Panel 1: Terminal price distribution ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(S_T, bins=120, color=BLUE, alpha=0.75, edgecolor="none")
    ax1.axvline(K,        color=ORANGE, lw=1.5, linestyle="--", label=f"Strike K={K}")
    ax1.axvline(S_T.mean(), color=GREEN,  lw=1.5, linestyle="-",  label=f"Mean S_T={S_T.mean():.2f}")
    ax1.set_title("GBM Terminal Price Distribution", color="#e6edf3")
    ax1.set_xlabel("Terminal Stock Price  S_T")
    ax1.set_ylabel("Frequency")
    ax1.legend(fontsize=8)
    ax1.grid(True)

    # ── Panel 2: Convergence analysis ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    sim_counts, mc_prices, ci_lo, ci_hi = convergence_analysis(
        S, K, T, r, sigma, option_type)
    ax2.fill_between(sim_counts, ci_lo, ci_hi, alpha=0.25, color=BLUE, label="95% CI")
    ax2.plot(sim_counts, mc_prices, color=BLUE,   lw=1.5, label="MC Price")
    ax2.axhline(bs_price,          color=ORANGE,  lw=1.5, linestyle="--", label=f"BS Price={bs_price:.4f}")
    ax2.set_xscale("log")
    ax2.set_title("Convergence: MC → Black-Scholes", color="#e6edf3")
    ax2.set_xlabel("Number of Simulations")
    ax2.set_ylabel("Option Price")
    ax2.legend(fontsize=8)
    ax2.grid(True)

    # ── Panel 3: Call price vs spot ────────────────────────────────────────────
    ax3   = fig.add_subplot(gs[1, 0])
    spots = np.linspace(60, 150, 40)
    mc_calls = [monte_carlo_price(s, K, T, r, sigma, "call", 50_000, seed=i)[0]
                for i, s in enumerate(spots)]
    bs_calls = [black_scholes(s, K, T, r, sigma, "call") for s in spots]
    ax3.plot(spots, bs_calls, color=ORANGE, lw=2,   label="Black-Scholes")
    ax3.scatter(spots, mc_calls, color=BLUE, s=18,  label="Monte Carlo",  zorder=3)
    ax3.axvline(K, color="#8b949e", lw=1, linestyle=":")
    ax3.set_title("Call Price vs Spot Price", color="#e6edf3")
    ax3.set_xlabel("Spot Price  S")
    ax3.set_ylabel("Call Price")
    ax3.legend(fontsize=8)
    ax3.grid(True)

    # ── Panel 4: Put price vs spot ─────────────────────────────────────────────
    ax4   = fig.add_subplot(gs[1, 1])
    mc_puts = [monte_carlo_price(s, K, T, r, sigma, "put", 50_000, seed=i)[0]
               for i, s in enumerate(spots)]
    bs_puts = [black_scholes(s, K, T, r, sigma, "put") for s in spots]
    ax4.plot(spots, bs_puts, color=ORANGE, lw=2,   label="Black-Scholes")
    ax4.scatter(spots, mc_puts, color=PURPLE, s=18, label="Monte Carlo", zorder=3)
    ax4.axvline(K, color="#8b949e", lw=1, linestyle=":")
    ax4.set_title("Put Price vs Spot Price", color="#e6edf3")
    ax4.set_xlabel("Spot Price  S")
    ax4.set_ylabel("Put Price")
    ax4.legend(fontsize=8)
    ax4.grid(True)

    # ── Panel 5: Price vs volatility ───────────────────────────────────────────
    ax5    = fig.add_subplot(gs[2, 0])
    sigmas = np.linspace(0.05, 0.8, 50)
    bs_vol_call = [black_scholes(S, K, T, r, s, "call") for s in sigmas]
    bs_vol_put  = [black_scholes(S, K, T, r, s, "put")  for s in sigmas]
    ax5.plot(sigmas * 100, bs_vol_call, color=BLUE,   lw=2, label="Call")
    ax5.plot(sigmas * 100, bs_vol_put,  color=PURPLE,  lw=2, label="Put")
    ax5.axvline(sigma * 100, color="#8b949e", lw=1, linestyle=":", label=f"σ={sigma*100:.0f}%")
    ax5.set_title("Option Price vs Volatility", color="#e6edf3")
    ax5.set_xlabel("Volatility  σ (%)")
    ax5.set_ylabel("Option Price")
    ax5.legend(fontsize=8)
    ax5.grid(True)

    # ── Panel 6: Price vs time to expiry ──────────────────────────────────────
    ax6   = fig.add_subplot(gs[2, 1])
    times = np.linspace(0.02, 2.0, 50)
    bs_t_call = [black_scholes(S, K, t, r, sigma, "call") for t in times]
    bs_t_put  = [black_scholes(S, K, t, r, sigma, "put")  for t in times]
    ax6.plot(times, bs_t_call, color=BLUE,   lw=2, label="Call")
    ax6.plot(times, bs_t_put,  color=PURPLE,  lw=2, label="Put")
    ax6.axvline(T, color="#8b949e", lw=1, linestyle=":", label=f"T={T}yr")
    ax6.set_title("Option Price vs Time to Expiry", color="#e6edf3")
    ax6.set_xlabel("Time to Expiry  T (years)")
    ax6.set_ylabel("Option Price")
    ax6.legend(fontsize=8)
    ax6.grid(True)

    # ── Panel 7: Greeks vs spot price ─────────────────────────────────────────
    ax7    = fig.add_subplot(gs[3, 0])
    greeks = {k: [] for k in ["delta", "gamma", "vega", "theta"]}
    for s in spots:
        g = bs_greeks(s, K, T, r, sigma, option_type)
        for key in greeks:
            greeks[key].append(g[key])
    colours = [BLUE, GREEN, ORANGE, PURPLE]
    for (name, vals), col in zip(greeks.items(), colours):
        ax7.plot(spots, vals, color=col, lw=1.8, label=name.capitalize())
    ax7.axvline(K, color="#8b949e", lw=1, linestyle=":")
    ax7.axhline(0, color="#8b949e", lw=0.8)
    ax7.set_title("Greeks vs Spot Price", color="#e6edf3")
    ax7.set_xlabel("Spot Price  S")
    ax7.set_ylabel("Greek Value")
    ax7.legend(fontsize=8)
    ax7.grid(True)

    # ── Panel 8: Implied volatility smile ─────────────────────────────────────
    ax8     = fig.add_subplot(gs[3, 1])
    strikes = np.linspace(70, 135, 35)
    # Simulate market prices with a vol smile (skew + curvature)
    smile_vol   = 0.2 + 0.3 * ((strikes - S) / S) ** 2 - 0.05 * (strikes - S) / S
    mkt_prices  = [black_scholes(S, k, T, r, v, "call")
                   for k, v in zip(strikes, smile_vol)]
    implied_vols = [implied_volatility(p, S, k, T, r, "call") * 100
                    for p, k in zip(mkt_prices, strikes)]
    ax8.plot(strikes, implied_vols, color=YELLOW, lw=2, label="Implied Vol")
    ax8.axhline(sigma * 100, color="#8b949e", lw=1, linestyle="--",
                label=f"Flat σ={sigma*100:.0f}%")
    ax8.axvline(S, color=GREEN, lw=1, linestyle=":", label=f"ATM S={S}")
    ax8.set_title("Implied Volatility Smile", color="#e6edf3")
    ax8.set_xlabel("Strike Price  K")
    ax8.set_ylabel("Implied Volatility (%)")
    ax8.legend(fontsize=8)
    ax8.grid(True)

    plt.savefig("option_pricer_output.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"\nSaved → option_pricer_output.png")

    return mc_price, ci, bs_price


# ── Summary Print ─────────────────────────────────────────────────────────────
def print_summary(S=100, K=100, T=1.0, r=0.05, sigma=0.2,
                  option_type="call", n_simulations=100_000):
    mc_price, ci, bs_price = plot_results(S, K, T, r, sigma, option_type, n_simulations)
    greeks = bs_greeks(S, K, T, r, sigma, option_type)

    print("\n" + "═" * 52)
    print(f"  MONTE CARLO OPTION PRICER  ·  European {option_type.upper()}")
    print("═" * 52)
    print(f"  Parameters : S={S}, K={K}, T={T}yr, r={r*100:.1f}%, σ={sigma*100:.0f}%")
    print(f"  Simulations: {n_simulations:,}")
    print("─" * 52)
    print(f"  MC Price   : {mc_price:.4f}")
    print(f"  95% CI     : [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"  BS Price   : {bs_price:.4f}")
    print(f"  Error      : {abs(mc_price - bs_price):.4f}  "
          f"({abs(mc_price - bs_price)/bs_price*100:.3f}%)")
    print("─" * 52)
    print(f"  Δ Delta    : {greeks['delta']:+.4f}")
    print(f"  Γ Gamma    : {greeks['gamma']:+.4f}")
    print(f"  ν Vega     : {greeks['vega']:+.4f}  (per 1% Δσ)")
    print(f"  θ Theta    : {greeks['theta']:+.4f}  (per day)")
    print(f"  ρ Rho      : {greeks['rho']:+.4f}  (per 1% Δr)")
    print("═" * 52 + "\n")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_summary(
        S=100,           # Spot price
        K=100,           # Strike price
        T=1.0,           # Time to expiry (years)
        r=0.05,          # Risk-free rate
        sigma=0.20,      # Volatility
        option_type="call",
        n_simulations=100_000,
    )
