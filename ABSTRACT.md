# CIO-Agent FAB++: Dynamic Finance Agent Benchmark

**Team AgentBusters** | AgentBeats Competition 2026

## Abstract

CIO-Agent FAB++ (Finance Agent Benchmark Plus Plus) is a comprehensive Green Agent for evaluating AI finance agents on real-world financial analysis tasks. The benchmark dynamically generates evaluation tasks across **18 categories** spanning fundamental analysis, quantitative reasoning, options trading, and risk management.

### Evaluation Categories

**Core Finance Analysis:**
- **Beat or Miss**: Earnings surprise detection against analyst consensus
- **Macro Analysis**: Economic trend interpretation and market impact assessment
- **Fundamental Analysis**: Financial statement interpretation (10-K, 10-Q filings)
- **Quantitative Reasoning**: Numerical calculations from financial data
- **SEC Filing Analysis**: Information extraction from regulatory documents
- **Trend Analysis**: Historical pattern recognition and forecasting

**Options Alpha Challenge:**
- **Options Pricing**: Black-Scholes valuation and fair value assessment
- **Greeks Analysis**: Delta, gamma, theta, vega calculations and hedging
- **Strategy Construction**: Multi-leg options strategies (iron condors, spreads, straddles)
- **Volatility Trading**: IV rank/percentile analysis and volatility arbitrage
- **P&L Attribution**: Return decomposition by Greek exposure
- **Risk Management**: VaR-based position sizing and stress testing

**Advanced Scenarios:**
- **Copy Trading**: Strategy replication and signal generation
- **Race to 10M**: Capital growth optimization under constraints
- **Strategy Defense**: Adversarial robustness testing

### Evaluation Methodology

The Green Agent employs a multi-dimensional scoring system:

1. **Role-Based Scoring** (0-100):
   - Macro thesis quality and market insight
   - Fundamental accuracy against ground truth
   - Execution methodology and tool usage

2. **Options-Specific Scoring** (for options tasks):
   - P&L Accuracy (25%): Profit/loss calculations
   - Greeks Accuracy (25%): Sensitivity analysis
   - Strategy Quality (25%): Structure and rationale
   - Risk Management (25%): Position sizing and hedging

3. **Adversarial Debate**: Counter-argument generation to test conviction, with debate multipliers (0.8x - 1.2x) based on rebuttal quality

4. **Alpha Score**: Final composite score combining role performance, debate robustness, and cost efficiency

### Data Infrastructure

The benchmark integrates with **6 MCP (Model Context Protocol) servers**:

| Server | Function |
|--------|----------|
| SEC EDGAR | Real-time SEC filing access with temporal locking |
| Yahoo Finance | Market data with lookahead detection |
| Python Sandbox | Secure code execution for calculations |
| Options Chain | Black-Scholes pricing and Greeks |
| Trading Simulator | Paper trading with realistic slippage |
| Risk Metrics | VaR, Sharpe ratio, stress testing |

### Dataset Support

- **Synthetic Questions**: 537 dynamically generated tasks from financial data lake
- **BizFinBench v2**: HiThink benchmark for financial reasoning (EN/CN)
- **Public CSV**: Curated finance QA pairs with ground truth
- **Real-time Generation**: Tasks generated from live market data

### Key Features

- **Temporal Locking**: Prevents lookahead bias by locking data to simulation date
- **Dynamic Task Generation**: Creates novel tasks from templates with real financial data
- **A2A Protocol**: Full compliance with Agent-to-Agent communication standard
- **Cost Tracking**: Monitors LLM token usage and tool call expenses
- **Configurable Evaluation**: YAML-based multi-dataset configuration

### Technical Specifications

- **Green Agent**: A2A server on port 9109
- **Purple Agent (Baseline)**: Finance analyst on port 9110
- **MCP Servers**: Ports 8101-8106
- **Supported Models**: OpenAI GPT-4o, Anthropic Claude, local vLLM

### Results

Baseline Purple Agent performance on sample tasks:
- Synthetic Questions: **Alpha Score 60,418**
- Public CSV: **66.67% accuracy**
- Options Tasks: **50.8/100 options score**

---

**Repository**: https://github.com/yxc20089/AgentBusters
**Docker Images**:
- `ghcr.io/yxc20089/agentbusters-green:latest`
- `ghcr.io/yxc20089/agentbusters-purple:latest`
