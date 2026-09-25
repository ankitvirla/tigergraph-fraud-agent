import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Brain,
  CheckCircle2,
  ChevronDown,
  CircleDot,
  Database,
  GitBranch,
  Layers3,
  Network,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Target,
  UserRound,
  Workflow,
  XCircle,
  Zap,
} from "lucide-react";
import "./App.css";

const API_URL = "http://localhost:8000";

const CASE_IDS = Array.from(
  { length: 20 },
  (_, i) => `HHG-${String(i + 1).padStart(3, "0")}`
);

const DEMO_CASES = [
  {
    id: "HHG-009",
    label: "High uncertainty",
    description: "Insufficient supporting evidence",
  },
  {
    id: "HHG-019",
    label: "Low probability",
    description: "5 independent evidence groups",
  },
  {
    id: "HHG-005",
    label: "Low uncertainty",
    description: "Strong evidence coverage",
  },
];

const formatProbability = (value) =>
  `${(Number(value || 0) * 100).toFixed(2)}%`;

const formatLabel = (value) =>
  String(value || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

function RiskBadge({ probability }) {
  const p = Number(probability || 0);

  let cls = "risk-high";
  let label = "HIGH";

  if (p < 0.35) {
    cls = "risk-low";
    label = "LOW";
  } else if (p < 0.75) {
    cls = "risk-medium";
    label = "MEDIUM";
  }

  return <span className={`risk-badge ${cls}`}>{label}</span>;
}

function UncertaintyBadge({ value }) {
  const cls =
    value === "low"
      ? "uncertainty-low"
      : value === "medium"
        ? "uncertainty-medium"
        : "uncertainty-high";

  return (
    <span className={`uncertainty-badge ${cls}`}>
      {String(value || "unknown").toUpperCase()}
    </span>
  );
}

function MetricCard({ icon: Icon, label, value, sub, accent }) {
  return (
    <div className={`metric-card ${accent || ""}`}>
      <div className="metric-icon">
        <Icon size={18} />
      </div>

      <div className="metric-content">
        <div className="metric-label">{label}</div>
        <div className="metric-value">{value}</div>

        {sub && <div className="metric-sub">{sub}</div>}
      </div>
    </div>
  );
}

function EvidenceItem({ item }) {
  const strength = item.strength || "not_material";
  const direction = item.direction || "context";

  const strengthClass =
    strength === "strong"
      ? "evidence-strong"
      : strength === "medium"
        ? "evidence-medium"
        : strength === "weak"
          ? "evidence-weak"
          : "evidence-neutral";

  const directionIcon =
    direction === "supports_fraud" ? (
      <ShieldAlert size={15} />
    ) : direction === "contradicts_fraud" ? (
      <XCircle size={15} />
    ) : (
      <CircleDot size={15} />
    );

  return (
    <div className={`evidence-item ${strengthClass}`}>
      <div className="evidence-icon">{directionIcon}</div>

      <div className="evidence-body">
        <div className="evidence-title">
          {formatLabel(item.signal_type || item.type || "Evidence")}
        </div>

        <div className="evidence-description">
          {item.explanation ||
            item.description ||
            item.reason ||
            "Evidence observed during investigation."}
        </div>

        <div className="evidence-meta">
          <span>{formatLabel(strength)}</span>
          <span>•</span>
          <span>{formatLabel(direction)}</span>

          {item.independence_group && (
            <>
              <span>•</span>
              <span>{formatLabel(item.independence_group)}</span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function EvidenceSection({ title, items, icon: Icon, emptyText }) {
  return (
    <div className="evidence-section">
      <div className="section-heading">
        <div className="section-heading-left">
          <Icon size={17} />
          <span>{title}</span>
        </div>

        <span className="section-count">{items.length}</span>
      </div>

      {items.length === 0 ? (
        <div className="empty-evidence">{emptyText}</div>
      ) : (
        <div className="evidence-list">
          {items.map((item, index) => (
            <EvidenceItem
              key={`${item.signal_type || item.type || "evidence"}-${index}`}
              item={item}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function TimelineStep({ number, title, description }) {
  return (
    <div className="timeline-step completed">
      <div className="timeline-number">{number}</div>

      <div className="timeline-dot">
        <CheckCircle2 size={15} />
      </div>

      <div>
        <div className="timeline-title">{title}</div>
        <div className="timeline-description">{description}</div>
      </div>
    </div>
  );
}

function FlowNode({ icon: Icon, title, value, tone = "" }) {
  return (
    <div className={`flow-node ${tone}`}>
      <div className="flow-node-icon">
        <Icon size={15} />
      </div>

      <div className="flow-node-title">{title}</div>
      <div className="flow-node-value">{value}</div>
    </div>
  );
}

function NetworkNode({ icon: Icon, label, value }) {
  return (
    <div className="network-node">
      <div className="network-node-icon">
        <Icon size={17} />
      </div>

      <div>
        <div className="network-node-label">{label}</div>
        <div className="network-node-value">{value}</div>
      </div>
    </div>
  );
}

function App() {
  const [selectedCase, setSelectedCase] = useState("HHG-019");
  const [caseData, setCaseData] = useState(null);
  const [benchmark, setBenchmark] = useState(null);

  const [loading, setLoading] = useState(true);
  const [runningAgent, setRunningAgent] = useState(false);
  const [agentError, setAgentError] = useState(null);

  const [showRaw, setShowRaw] = useState(false);
  const [showBenchmark, setShowBenchmark] = useState(false);
  const [lastRunLive, setLastRunLive] = useState(false);

  /*
   * Load benchmark/static result.
   */
  useEffect(() => {
    setLoading(true);
    setAgentError(null);

    Promise.all([
      fetch(`/data/${selectedCase}.json`).then((response) => {
        if (!response.ok) {
          throw new Error(`Unable to load ${selectedCase}`);
        }

        return response.json();
      }),

      fetch("/data/benchmark_summary.json")
        .then((response) => (response.ok ? response.json() : null))
        .catch(() => null),
    ])
      .then(([data, summary]) => {
        setCaseData(data);
        setBenchmark(summary);
        setLoading(false);
        setLastRunLive(false);
      })
      .catch((error) => {
        console.error(error);

        setCaseData(null);
        setLoading(false);
        setAgentError(error.message);
      });
  }, [selectedCase]);

  /*
   * Call the actual investigation agent.
   */
  async function runAgent() {
    if (runningAgent) {
      return;
    }

    setRunningAgent(true);
    setAgentError(null);
    setLastRunLive(false);

    try {
      const response = await fetch(
        `${API_URL}/api/investigate/${selectedCase}`,
        {
          method: "POST",
          headers: {
            Accept: "application/json",
          },
        }
      );

      if (!response.ok) {
        let message = `Agent failed with status ${response.status}`;

        try {
          const errorBody = await response.json();

          if (errorBody?.detail) {
            message = errorBody.detail;
          }
        } catch {
          // Keep default error message.
        }

        throw new Error(message);
      }

      const result = await response.json();

      setCaseData(result);
      setLastRunLive(true);
    } catch (error) {
      console.error("Agent investigation failed:", error);

      setAgentError(
        error?.message ||
          "Unable to connect to the investigation agent. Make sure the FastAPI backend is running on port 8000."
      );
    } finally {
      setRunningAgent(false);
    }
  }

  const evidence = caseData?.evidence || {};

  const allEvidence = useMemo(
    () => [
      ...(evidence.strong || []),
      ...(evidence.medium || []),
      ...(evidence.weak || []),
      ...(evidence.not_material || []),
      ...(evidence.supporting || []),
      ...(evidence.contradicting || []),
      ...(evidence.context || []),
    ],
    [evidence]
  );

  const uniqueEvidence = useMemo(() => {
    const seen = new Set();

    return allEvidence.filter((item) => {
      const key = JSON.stringify([
        item.signal_type,
        item.strength,
        item.direction,
        item.explanation,
      ]);

      if (seen.has(key)) {
        return false;
      }

      seen.add(key);
      return true;
    });
  }, [allEvidence]);

  const supporting = uniqueEvidence.filter(
    (item) => item.direction === "supports_fraud"
  );

  const contradicting = uniqueEvidence.filter(
    (item) => item.direction === "contradicts_fraud"
  );

  const contextual = uniqueEvidence.filter(
    (item) =>
      item.direction === "context" ||
      !item.direction ||
      !["supports_fraud", "contradicts_fraud"].includes(item.direction)
  );

  const facts = caseData?.graph_investigation?.facts || {};
  const risk = caseData?.risk_assessment || {};
  const uncertainty = caseData?.uncertainty || {};
  const nextAction = caseData?.next_action || {};
  const llm = caseData?.llm_explanation || {};

  const independentGroups =
    caseData?.evidence_summary?.independent_evidence_groups || [];

  const strongCount = caseData?.evidence_summary?.strong_count ?? 0;
  const mediumCount = caseData?.evidence_summary?.medium_count ?? 0;
  const weakCount = caseData?.evidence_summary?.weak_count ?? 0;

  const probability = Number(risk.fraud_probability || 0);

  const graphComplete =
    facts.customer_profile_success &&
    facts.transaction_context_success &&
    facts.shared_device_query_success;

  const actionText = formatLabel(
    nextAction.action || "review_evidence_and_network_context"
  );

  const summaryRows = Array.isArray(benchmark)
    ? benchmark
    : benchmark?.cases || benchmark?.results || [];

  const successfulCases = Array.isArray(benchmark)
    ? benchmark.filter((item) => item.status === "success").length
    : benchmark?.successful_cases ??
      benchmark?.success_count ??
      benchmark?.completed_cases ??
      20;

  const uncertaintyCounts = summaryRows.reduce(
    (acc, item) => {
      if (item.uncertainty === "high") acc.high += 1;
      if (item.uncertainty === "medium") acc.medium += 1;
      if (item.uncertainty === "low") acc.low += 1;

      return acc;
    },
    {
      high: 0,
      medium: 0,
      low: 0,
    }
  );

  const whyAssessment =
    probability < 0.35
      ? [
          `${independentGroups.length} independent evidence groups were identified.`,
          `${strongCount} strong and ${mediumCount} medium evidence signals were retained.`,
          graphComplete
            ? "TigerGraph context retrieval completed successfully."
            : "Graph retrieval was only partially successful.",
          "The calibrated model output remains below the low-risk presentation boundary.",
        ]
      : [
          `${independentGroups.length} independent evidence groups were identified.`,
          `${strongCount} strong and ${mediumCount} medium evidence signals were retained.`,
          graphComplete
            ? "TigerGraph context retrieval completed successfully."
            : "Graph retrieval was only partially successful.",
          contradicting.length
            ? `${contradicting.length} contradicting evidence item(s) require review.`
            : "No contradicting evidence was identified.",
        ];

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <ShieldCheck size={23} />
          </div>

          <div>
            <div className="brand-title">Fraud Investigation Agent</div>

            <div className="brand-subtitle">
              TigerGraph • MCP • Calibrated ML • Local Qwen
            </div>
          </div>
        </div>

        <div className="topbar-status">
          <span className="status-dot" />
          Investigation Engine Online
        </div>
      </header>

      <div className="app-layout">
        <aside className="sidebar">
          <div className="sidebar-heading">
            <div>
              <div className="sidebar-title">Investigation Cases</div>
              <div className="sidebar-subtitle">20 benchmark cases</div>
            </div>

            <Layers3 size={18} />
          </div>

          <div className="demo-block">
            <div className="demo-heading">
              <Sparkles size={13} />
              DEMO CASES
            </div>

            {DEMO_CASES.map((item) => (
              <button
                key={item.id}
                className={`demo-case ${
                  selectedCase === item.id ? "active" : ""
                }`}
                onClick={() => setSelectedCase(item.id)}
              >
                <div>
                  <strong>{item.id}</strong>
                  <span>{item.label}</span>
                </div>

                <ArrowRight size={14} />
              </button>
            ))}
          </div>

          <div className="case-divider" />

          <div className="case-list">
            {CASE_IDS.map((id) => {
              const active = id === selectedCase;

              return (
                <button
                  key={id}
                  className={`case-button ${active ? "active" : ""}`}
                  onClick={() => setSelectedCase(id)}
                >
                  <span>{id}</span>

                  {active && <ArrowRight size={15} />}
                </button>
              );
            })}
          </div>

          <div className="sidebar-footer">
            <div className="system-item">
              <span className="mini-dot green" />
              TigerGraph MCP
            </div>

            <div className="system-item">
              <span className="mini-dot green" />
              Evidence Calibration
            </div>

            <div className="system-item">
              <span className="mini-dot green" />
              Qwen Reasoner
            </div>
          </div>
        </aside>

        <main className="main-content">
          {loading ? (
            <div className="loading-screen">
              <div className="spinner" />
              Loading investigation...
            </div>
          ) : !caseData ? (
            <div className="error-card">
              Unable to load {selectedCase}
            </div>
          ) : (
            <>
              <div className="page-header">
                <div className="page-header-copy">
                  <div className="eyebrow">
                    INVESTIGATION / {caseData.case_id}
                  </div>

                  <h1>
                    Case {caseData.case_id}

                    <span className="complete-badge">
                      <CheckCircle2 size={14} />
                      Investigation Complete
                    </span>
                  </h1>

                  <p>
                    Graph-grounded fraud investigation combining TigerGraph
                    context, calibrated evidence, probability estimation and
                    local LLM reasoning.
                  </p>
                </div>

                <button
                  className={`run-agent-button ${
                    runningAgent ? "running" : ""
                  }`}
                  onClick={runAgent}
                  disabled={runningAgent}
                >
                  {runningAgent ? (
                    <>
                      <div className="button-spinner" />
                      Agent Investigating...
                    </>
                  ) : (
                    <>
                      <Zap size={16} />
                      {lastRunLive
                        ? "Re-run Agent Investigation"
                        : "Run Agent Investigation"}
                    </>
                  )}
                </button>
              </div>

              {runningAgent && (
                <div className="agent-running-banner">
                  <div className="button-spinner" />

                  <div>
                    <strong>Agent investigation running</strong>

                    <span>
                      TigerGraph MCP → evidence calibration → probability model
                      → Qwen
                    </span>
                  </div>
                </div>
              )}

              {lastRunLive && !runningAgent && (
                <div className="agent-success-banner">
                  <CheckCircle2 size={15} />

                  <div>
                    <strong>Live agent investigation completed</strong>

                    <span>
                      This dashboard is displaying the freshly generated agent
                      result.
                    </span>
                  </div>

                  <span className="live-pill">LIVE RESULT</span>
                </div>
              )}

              {agentError && (
                <div className="agent-error-banner">
                  <AlertTriangle size={15} />
                  <span>{agentError}</span>
                </div>
              )}

              <section className="metrics-grid">
                <MetricCard
                  icon={Target}
                  label="Fraud Probability"
                  value={formatProbability(risk.fraud_probability)}
                  sub="Calibrated case-level estimate"
                  accent="metric-risk"
                />

                <MetricCard
                  icon={AlertTriangle}
                  label="Uncertainty"
                  value={
                    <UncertaintyBadge
                      value={uncertainty.level || "unknown"}
                    />
                  }
                  sub={`${independentGroups.length} independent evidence group${
                    independentGroups.length === 1 ? "" : "s"
                  }`}
                />

                <MetricCard
                  icon={Activity}
                  label="Evidence"
                  value={strongCount + mediumCount + weakCount}
                  sub={`${strongCount} strong • ${mediumCount} medium • ${weakCount} weak`}
                />

                <MetricCard
                  icon={Network}
                  label="Graph Network"
                  value={facts.shared_customer_count ?? 0}
                  sub={`${facts.device_count ?? 0} connected device${
                    facts.device_count === 1 ? "" : "s"
                  }`}
                />
              </section>

              <section className="panel reasoning-panel">
                <div className="panel-header">
                  <div>
                    <div className="panel-kicker">
                      AGENT REASONING PIPELINE
                    </div>

                    <h2>How the Investigation Reached This Assessment</h2>
                  </div>

                  <Workflow size={18} />
                </div>

                <div className="flow">
                  <FlowNode
                    icon={Network}
                    title="Graph"
                    value="TigerGraph context"
                    tone="blue"
                  />

                  <div className="flow-arrow">→</div>

                  <FlowNode
                    icon={ShieldCheck}
                    title="Evidence"
                    value={`${supporting.length} supporting`}
                    tone="green"
                  />

                  <div className="flow-arrow">→</div>

                  <FlowNode
                    icon={GitBranch}
                    title="Independence"
                    value={`${independentGroups.length} groups`}
                    tone="purple"
                  />

                  <div className="flow-arrow">→</div>

                  <FlowNode
                    icon={BarChart3}
                    title="Model"
                    value="18 features"
                    tone="amber"
                  />

                  <div className="flow-arrow">→</div>

                  <FlowNode
                    icon={Target}
                    title="Probability"
                    value={formatProbability(probability)}
                    tone="red"
                  />

                  <div className="flow-arrow">→</div>

                  <FlowNode
                    icon={AlertTriangle}
                    title="Uncertainty"
                    value={formatLabel(uncertainty.level)}
                    tone="amber"
                  />

                  <div className="flow-arrow">→</div>

                  <FlowNode
                    icon={ArrowRight}
                    title="Action"
                    value={actionText}
                    tone="blue"
                  />
                </div>

                <div className="reasoning-note">
                  <Brain size={15} />

                  <span>
                    Deterministic graph, evidence and probability outputs remain
                    authoritative; Qwen is used to explain the grounded
                    result.
                  </span>
                </div>
              </section>

              <div className="content-grid">
                <div className="left-column">
                  <section className="panel case-overview">
                    <div className="panel-header">
                      <div>
                        <div className="panel-kicker">CASE OVERVIEW</div>
                        <h2>Investigation Target</h2>
                      </div>

                      <RiskBadge probability={risk.fraud_probability} />
                    </div>

                    <div className="target-grid">
                      <div className="target-item">
                        <span>Customer</span>
                        <strong>{caseData.case.customer_id}</strong>
                      </div>

                      <div className="target-item">
                        <span>Transaction</span>
                        <strong>{caseData.case.flagged_txn_id}</strong>
                      </div>

                      <div className="target-item">
                        <span>Card</span>
                        <strong>{caseData.case.card_id}</strong>
                      </div>

                      <div className="target-item">
                        <span>Amount</span>
                        <strong>
                          ${Number(caseData.case.amount || 0).toFixed(2)}
                        </strong>
                      </div>

                      <div className="target-item">
                        <span>Channel</span>
                        <strong>{caseData.case.channel}</strong>
                      </div>

                      <div className="target-item">
                        <span>Product</span>
                        <strong>{caseData.case.ProductCD}</strong>
                      </div>

                      <div className="target-item">
                        <span>Trigger</span>
                        <strong>
                          {formatLabel(caseData.case.trigger_type)}
                        </strong>
                      </div>

                      <div className="target-item">
                        <span>Risk Score</span>
                        <strong>{caseData.case.risk_score}</strong>
                      </div>
                    </div>
                  </section>

                  <section className="panel">
                    <div className="panel-header">
                      <div>
                        <div className="panel-kicker">AGENT TRACE</div>
                        <h2>Investigation Progression</h2>
                      </div>

                      <Zap size={18} />
                    </div>

                    <div className="timeline">
                      <TimelineStep
                        number="01"
                        title="Customer context retrieved"
                        description="TigerGraph customer profile, cards, devices, cases and transactions"
                      />

                      <TimelineStep
                        number="02"
                        title="Transaction context retrieved"
                        description="Transaction, related customer, card, device and containing cases"
                      />

                      <TimelineStep
                        number="03"
                        title="Network relationships inspected"
                        description="Shared-device customer network retrieved through graph traversal"
                      />

                      <TimelineStep
                        number="04"
                        title="Evidence calibrated"
                        description="Raw observations converted into strength, direction and independence"
                      />

                      <TimelineStep
                        number="05"
                        title="Probability calculated"
                        description="18-feature HistGradientBoosting model with Platt calibration"
                      />

                      <TimelineStep
                        number="06"
                        title="Uncertainty assessed"
                        description={`Current assessment: ${formatLabel(
                          uncertainty.level
                        )}`}
                      />

                      <TimelineStep
                        number="07"
                        title="Next action selected"
                        description={actionText}
                      />
                    </div>
                  </section>

                  <section className="panel">
                    <div className="panel-header">
                      <div>
                        <div className="panel-kicker">
                          GRAPH INTELLIGENCE
                        </div>
                        <h2>Relationship Context</h2>
                      </div>

                      <GitBranch size={18} />
                    </div>

                    <div className="graph-visual">
                      <div className="graph-center">
                        <div className="graph-avatar">
                          <UserRound size={20} />
                        </div>

                        <strong>{caseData.case.customer_id}</strong>
                        <span>Customer</span>
                      </div>

                      <div className="graph-orbit orbit-top">
                        <div className="graph-link" />

                        <div className="graph-pill">
                          <Database size={14} />
                          <span>{facts.device_count ?? 0} Devices</span>
                        </div>
                      </div>

                      <div className="graph-orbit orbit-left">
                        <div className="graph-link" />

                        <div className="graph-pill">
                          <Network size={14} />
                          <span>
                            {facts.shared_customer_count ?? 0} Related
                          </span>
                        </div>
                      </div>

                      <div className="graph-orbit orbit-right">
                        <div className="graph-link" />

                        <div className="graph-pill">
                          <ShieldAlert size={14} />
                          <span>
                            {facts.historical_case_count ?? 0} Cases
                          </span>
                        </div>
                      </div>

                      <div className="graph-orbit orbit-bottom">
                        <div className="graph-link" />

                        <div className="graph-pill">
                          <Database size={14} />
                          <span>Transaction</span>
                        </div>
                      </div>
                    </div>

                    <div className="network-grid">
                      <NetworkNode
                        icon={UserRound}
                        label="Customer"
                        value={caseData.case.customer_id}
                      />

                      <NetworkNode
                        icon={Database}
                        label="Devices"
                        value={facts.device_count ?? 0}
                      />

                      <NetworkNode
                        icon={Network}
                        label="Related Customers"
                        value={facts.shared_customer_count ?? 0}
                      />

                      <NetworkNode
                        icon={ShieldAlert}
                        label="Historical Cases"
                        value={facts.historical_case_count ?? 0}
                      />
                    </div>

                    <div className="graph-note">
                      <Network size={16} />

                      <span>
                        Relationships are retrieved from TigerGraph through
                        MCP. The LLM does not invent graph connections.
                      </span>
                    </div>
                  </section>

                  <section className="panel">
                    <div className="panel-header">
                      <div>
                        <div className="panel-kicker">EVIDENCE</div>
                        <h2>Investigation Evidence</h2>
                      </div>

                      <div className="evidence-summary">
                        <span>{supporting.length} supporting</span>
                        <span>{contradicting.length} contradicting</span>
                        <span>{contextual.length} contextual</span>
                      </div>
                    </div>

                    <EvidenceSection
                      title="Supporting Evidence"
                      items={supporting}
                      icon={ShieldCheck}
                      emptyText="No supporting evidence identified."
                    />

                    <EvidenceSection
                      title="Contradicting Evidence"
                      items={contradicting}
                      icon={XCircle}
                      emptyText="No contradicting evidence identified."
                    />

                    <EvidenceSection
                      title="Contextual Evidence"
                      items={contextual}
                      icon={CircleDot}
                      emptyText="No contextual evidence identified."
                    />
                  </section>
                </div>

                <div className="right-column">
                  <section className="panel risk-panel">
                    <div className="panel-kicker">RISK ASSESSMENT</div>

                    <div className="risk-number">
                      {formatProbability(risk.fraud_probability)}
                    </div>

                    <div className="risk-caption">
                      Calibrated historical case-level fraud probability
                    </div>

                    <div className="probability-track">
                      <div
                        className="probability-fill"
                        style={{
                          width: `${Math.min(100, probability * 100)}%`,
                        }}
                      />
                    </div>

                    <div className="risk-details">
                      <div>
                        <span>Model</span>
                        <strong>
                          HistGradientBoosting + Platt Calibration
                        </strong>
                      </div>

                      <div>
                        <span>Features</span>
                        <strong>{risk.feature_count || 18}</strong>
                      </div>

                      <div>
                        <span>Calibration</span>
                        <strong>{formatLabel(risk.calibration)}</strong>
                      </div>
                    </div>
                  </section>

                  <section className="panel why-panel">
                    <div className="panel-kicker">WHY THIS ASSESSMENT?</div>

                    <h2>Evidence → Decision Trace</h2>

                    <div className="why-list">
                      {whyAssessment.map((item, index) => (
                        <div className="why-item" key={index}>
                          <CheckCircle2 size={15} />
                          <span>{item}</span>
                        </div>
                      ))}
                    </div>

                    <div className="why-result">
                      <div>
                        <span>Result</span>
                        <strong>{formatProbability(probability)}</strong>
                      </div>

                      <div>
                        <span>Uncertainty</span>
                        <UncertaintyBadge value={uncertainty.level} />
                      </div>
                    </div>
                  </section>

                  <section className="panel action-panel">
                    <div className="panel-kicker">NEXT ACTION</div>

                    <div className="action-icon">
                      <ArrowRight size={22} />
                    </div>

                    <h2>{actionText}</h2>

                    <p>
                      {nextAction.reason ||
                        "Review the available evidence and graph context before proceeding."}
                    </p>

                    <div className="action-status">
                      <span>Agent recommendation</span>
                      <span className="recommended">READY</span>
                    </div>
                  </section>

                  <section className="panel uncertainty-panel">
                    <div className="panel-header">
                      <div>
                        <div className="panel-kicker">UNCERTAINTY</div>
                        <h2>Confidence Assessment</h2>
                      </div>

                      <UncertaintyBadge value={uncertainty.level} />
                    </div>

                    <p>
                      {uncertainty.reason ||
                        "Uncertainty is determined from evidence coverage, independent evidence groups and contradictions."}
                    </p>

                    <div className="uncertainty-factors">
                      <div>
                        <span>Independent groups</span>
                        <strong>{independentGroups.length}</strong>
                      </div>

                      <div>
                        <span>Contradictions</span>
                        <strong>{contradicting.length}</strong>
                      </div>

                      <div>
                        <span>Graph retrieval</span>
                        <strong>
                          {graphComplete ? "Complete" : "Partial"}
                        </strong>
                      </div>
                    </div>
                  </section>

                  <section className="panel ai-panel">
                    <div className="ai-heading">
                      <div className="ai-icon">
                        <Sparkles size={18} />
                      </div>

                      <div>
                        <div className="panel-kicker">LOCAL QWEN</div>
                        <h2>AI Investigator</h2>
                      </div>
                    </div>

                    <div className="ai-summary">
                      {llm.summary ||
                        "The local reasoning model explanation is unavailable for this case."}
                    </div>

                    {Array.isArray(llm.key_findings) &&
                      llm.key_findings.length > 0 && (
                        <div className="ai-findings">
                          <div className="ai-subheading">Key Findings</div>

                          {llm.key_findings.map((finding, index) => (
                            <div className="ai-finding" key={index}>
                              <CheckCircle2 size={15} />
                              <span>{finding}</span>
                            </div>
                          ))}
                        </div>
                      )}

                    {llm.uncertainty_explanation && (
                      <div className="ai-block">
                        <div className="ai-subheading">
                          Uncertainty Explanation
                        </div>

                        <p>{llm.uncertainty_explanation}</p>
                      </div>
                    )}

                    {llm.recommended_next_step && (
                      <div className="ai-block">
                        <div className="ai-subheading">
                          Recommended Next Step
                        </div>

                        <p>{llm.recommended_next_step}</p>
                      </div>
                    )}

                    <div className="ai-footer">
                      <Brain size={14} />
                      Grounded only in deterministic investigation results
                    </div>
                  </section>

                  <section className="panel benchmark-panel">
                    <button
                      className="benchmark-header"
                      onClick={() => setShowBenchmark((value) => !value)}
                    >
                      <div>
                        <div className="panel-kicker">BENCHMARK</div>
                        <h2>20-Case Investigation Run</h2>
                      </div>

                      <ChevronDown
                        className={showBenchmark ? "rotate" : ""}
                        size={17}
                      />
                    </button>

                    <div className="benchmark-summary">
                      <div className="benchmark-big">
                        <strong>{successfulCases}/20</strong>
                        <span>cases completed</span>
                      </div>

                      <div className="benchmark-bars">
                        <div>
                          <span>High uncertainty</span>
                          <strong>{uncertaintyCounts.high}</strong>
                        </div>

                        <div>
                          <span>Medium uncertainty</span>
                          <strong>{uncertaintyCounts.medium}</strong>
                        </div>

                        <div>
                          <span>Low uncertainty</span>
                          <strong>{uncertaintyCounts.low}</strong>
                        </div>
                      </div>
                    </div>

                    {showBenchmark && (
                      <div className="benchmark-table">
                        {summaryRows.length > 0 ? (
                          summaryRows.map((item) => (
                            <button
                              key={item.case_id}
                              className="benchmark-row"
                              onClick={() => {
                                setSelectedCase(item.case_id);
                                setShowBenchmark(false);
                              }}
                            >
                              <span>{item.case_id}</span>

                              <span>
                                {formatProbability(item.fraud_probability)}
                              </span>

                              <UncertaintyBadge value={item.uncertainty} />

                              <ArrowRight size={13} />
                            </button>
                          ))
                        ) : (
                          <div className="benchmark-fallback">
                            All 20 generated case outputs are available in
                            submission/outputs.
                          </div>
                        )}
                      </div>
                    )}
                  </section>

                  <button
                    className="raw-toggle"
                    onClick={() => setShowRaw((value) => !value)}
                  >
                    {showRaw
                      ? "Hide raw investigation JSON"
                      : "View raw investigation JSON"}

                    <ChevronDown
                      size={15}
                      className={showRaw ? "rotate" : ""}
                    />
                  </button>

                  {showRaw && (
                    <pre className="raw-json">
                      {JSON.stringify(caseData, null, 2)}
                    </pre>
                  )}
                </div>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;