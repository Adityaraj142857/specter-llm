"""
app.py  —  Specter LLM  |  College Constitution Analyser

Four pages:
    1. 📄 Ingest Constitution   — upload PDF, run full pipeline
    2. 💬 Ask a Question        — RAG query over the constitution
    3. ⚠️  Risk Analysis         — governance risk flags across all clauses
    4. 🔗 New Clause Impact     — analyse a new clause before adding it

All heavy work is done in the backend modules.
Session state keys:
    clauses         : list[dict]   — ingested clause dicts
    graph           : nx.DiGraph   — clause relationship graph
    risk_results    : list[dict]   — cached risk analysis output
    ingested        : bool         — whether constitution is loaded
"""

import streamlit as st
import networkx as nx
import pandas as pd
import streamlit.components.v1 as components
from pyvis.network import Network

# ── page config must be first Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="Specter LLM",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── lazy imports (only import heavy modules when needed) ──────────────────
from ingestion.pdf_reader import ingest_constitution
from ingestion.embedder import embed_query
from storage.vector_store import VectorStore
from graph.builder import load_graph
from graph.traversal import get_impact_subgraph, score_emoji, score_label
from query.risk import analyse_risks, risk_summary
from query.impact import analyse_new_clause
from query.qa import answer_question          # existing module — kept as-is


# ── session state defaults ────────────────────────────────────────────────
def _init_state():
    defaults = {
        "clauses":      [],
        "graph":        None,
        "risk_results": [],
        "ingested":     False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── sidebar navigation ────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/scales.png", width=48)
    st.title("Specter LLM")
    st.caption("College Constitution Analyser")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📄 Ingest Constitution",
         "💬 Ask a Question",
         "⚠️ Risk Analysis",
         "🔗 New Clause Impact",
         "🕸️ Graph Insights"],
        label_visibility="collapsed",
    )

    st.divider()
    if st.session_state.ingested:
        st.success(f"✅ {len(st.session_state.clauses)} clauses loaded")
        if st.session_state.graph:
            G = st.session_state.graph
            st.caption(
                f"Graph: {G.number_of_nodes()} nodes · {G.number_of_edges()} edges"
            )
    else:
        st.warning("No constitution loaded yet.")


# =============================================================================
# PAGE 1 — Ingest Constitution
# =============================================================================
if page == "📄 Ingest Constitution":
    st.header("📄 Ingest Constitution")
    st.write(
        "Upload your college constitution PDF. "
        "The system will extract clauses, embed them, build the knowledge graph, "
        "and prepare all three analysis engines."
    )

    uploaded = st.file_uploader(
        "Choose a PDF file", type=["pdf"], key="pdf_upload"
    )

    if uploaded:
        st.info(f"File ready: **{uploaded.name}** ({uploaded.size:,} bytes)")

        col1, col2 = st.columns([1, 4])
        with col1:
            run = st.button("🚀 Run Ingestion", type="primary", use_container_width=True)
        with col2:
            st.caption(
                "This will: extract clauses → embed with nomic-embed-text → "
                "store in ChromaDB → build NetworkX graph with LLM-classified edges. "
                "Takes 1–5 minutes depending on document size."
            )

        if run:
            pdf_bytes = uploaded.read()

            with st.status("Running ingestion pipeline...", expanded=True) as status:
                st.write("📖 Extracting text from PDF...")
                try:
                    st.write("✂️  Splitting into clauses (hybrid rule + LLM)...")
                    st.write("🔢 Embedding clauses with nomic-embed-text...")
                    st.write("🗄️  Storing in ChromaDB...")
                    st.write("🕸️  Building clause graph with LLM edge classification...")

                    clauses, G = ingest_constitution(pdf_bytes)

                    st.session_state.clauses  = clauses
                    st.session_state.graph    = G
                    st.session_state.ingested = True
                    st.session_state.risk_results = []   # clear cached risks

                    status.update(label="✅ Ingestion complete!", state="complete")

                except Exception as e:
                    status.update(label="❌ Ingestion failed", state="error")
                    st.error(f"Error: {e}")
                    st.stop()

    # ── Results preview ───────────────────────────────────────────────────
    if st.session_state.ingested and st.session_state.clauses:
        st.divider()
        st.subheader("Clause Preview")

        clauses = st.session_state.clauses
        preview_df = pd.DataFrame([
            {
                "ID":      c["id"],
                "Section": c.get("section", ""),
                "Heading": c.get("heading", ""),
                "Length":  len(c.get("text", "")),
            }
            for c in clauses
        ])
        st.dataframe(preview_df, use_container_width=True, height=300)

        st.subheader("Graph Overview")
        G = st.session_state.graph
        if G:
            edge_types = {}
            for _, _, d in G.edges(data=True):
                t = d.get("type", "UNKNOWN")
                edge_types[t] = edge_types.get(t, 0) + 1

            cols = st.columns(len(edge_types) + 2)
            cols[0].metric("Clause nodes", G.number_of_nodes())
            cols[1].metric("Total edges", G.number_of_edges())
            for i, (etype, count) in enumerate(edge_types.items()):
                cols[i + 2].metric(etype, count)

            # Adjacency table for selected clause
            st.subheader("Explore a clause's connections")
            clause_ids = [c["id"] for c in clauses]
            selected = st.selectbox("Select clause", clause_ids)
            if selected and G.has_node(selected):
                node = G.nodes[selected]
                st.markdown(f"**{node.get('heading', selected)}**")
                st.write(node.get("text", ""))

                edges_out = [
                    {
                        "Target":   v,
                        "Target heading": G.nodes[v].get("heading", v),
                        "Type":     d.get("type"),
                        "Score":    d.get("score", 0),
                        "Strength": score_label(d.get("score", 0)),
                        "Reason":   d.get("reason", ""),
                    }
                    for _, v, d in G.out_edges(selected, data=True)
                ]
                if edges_out:
                    st.dataframe(pd.DataFrame(edges_out), use_container_width=True)
                else:
                    st.caption("No outgoing edges for this clause.")


# =============================================================================
# PAGE 2 — Ask a Question (RAG)
# =============================================================================
elif page == "💬 Ask a Question":
    st.header("💬 Ask a Question")

    if not st.session_state.ingested:
        st.warning("Please ingest a constitution first (page 1).")
        st.stop()

    st.write(
        "Ask anything about the constitution. "
        "The system retrieves the most relevant clauses and uses the graph "
        "to expand context before answering."
    )

    query = st.text_area(
        "Your question",
        placeholder="e.g. What are the quorum requirements for a general body meeting?",
        height=100,
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        top_k = st.slider("Clauses to retrieve", 3, 10, 5)
    with col2:
        use_graph = st.toggle(
            "Expand with graph neighbours", value=True,
            help="After vector retrieval, also include clauses that are connected "
                 "by DEPENDS_ON or CONFLICTS_WITH edges for richer context."
        )

    if st.button("🔍 Ask", type="primary") and query.strip():
        with st.spinner("Retrieving relevant clauses and generating answer..."):
            try:
                vs = VectorStore()
                G  = st.session_state.graph

                # Embed query and retrieve top-k clauses
                q_embedding = embed_query(query)
                results     = vs.query(q_embedding, top_k=top_k)

                # Graph expansion: add connected clauses
                if use_graph and G:
                    extra_ids = set()
                    for r in results:
                        cid = r["id"]
                        if G.has_node(cid):
                            for _, v, d in G.out_edges(cid, data=True):
                                if d.get("type") in ("DEPENDS_ON", "CONFLICTS_WITH"):
                                    extra_ids.add(v)
                    for eid in extra_ids:
                        if eid not in {r["id"] for r in results}:
                            node = G.nodes.get(eid, {})
                            results.append({
                                "id":      eid,
                                "heading": node.get("heading", ""),
                                "text":    node.get("text", ""),
                                "section": node.get("section", ""),
                                "distance": 0.5,   # graph-expanded, not similarity-ranked
                            })

                # Build context string
                context = "\n\n---\n\n".join(
                    f"[{r['heading']}]\n{r['text']}" for r in results
                )

                # Get answer from LLM
                answer = answer_question(query, context)

                # Display
                st.subheader("Answer")
                st.write(answer)

                with st.expander(f"📚 Source clauses used ({len(results)})"):
                    for r in results:
                        dist = r.get("distance", None)
                        tag  = f"(similarity: {1-dist:.2f})" if dist is not None and dist <= 0.5 else "(graph-expanded)"
                        st.markdown(f"**{r['heading']}** {tag}")
                        st.caption(r["text"][:300] + ("..." if len(r["text"]) > 300 else ""))
                        st.divider()

            except Exception as e:
                st.error(f"Error: {e}")


# =============================================================================
# PAGE 3 — Risk Analysis
# =============================================================================
elif page == "⚠️ Risk Analysis":
    st.header("⚠️ Governance Risk Analysis")

    if not st.session_state.ingested:
        st.warning("Please ingest a constitution first (page 1).")
        st.stop()

    st.write(
        "Analyses every clause for governance structure problems — "
        "power concentration, missing quorum rules, vague enforcement, and more."
    )

    # Run or use cached results
    if not st.session_state.risk_results:
        if st.button("🔍 Run Risk Analysis", type="primary"):
            clauses = st.session_state.clauses
            with st.status(f"Analysing {len(clauses)} clauses...", expanded=True) as status:
                try:
                    risks = analyse_risks(clauses)
                    st.session_state.risk_results = risks
                    status.update(label=f"✅ Found {len(risks)} risk flags", state="complete")
                except Exception as e:
                    status.update(label="❌ Analysis failed", state="error")
                    st.error(str(e))
                    st.stop()
    else:
        st.success(f"Showing cached results — {len(st.session_state.risk_results)} flags found.")
        if st.button("🔄 Re-run Analysis"):
            st.session_state.risk_results = []
            st.rerun()

    risks = st.session_state.risk_results
    if not risks:
        st.stop()

    # ── Summary metrics ───────────────────────────────────────────────────
    summary = risk_summary(risks)
    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total flags",   summary["total"])
    c2.metric("🔴 HIGH",       summary["HIGH"])
    c3.metric("🟡 MEDIUM",     summary["MEDIUM"])
    c4.metric("🟢 LOW",        summary["LOW"])

    # Category breakdown
    with st.expander("By category"):
        cat_df = pd.DataFrame([
            {"Category": k, "Count": v}
            for k, v in summary["by_category"].items()
            if v > 0
        ]).sort_values("Count", ascending=False)
        st.dataframe(cat_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Filters ───────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        sev_filter = st.multiselect(
            "Filter by severity",
            ["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM"],
        )
    with col2:
        CATEGORIES = [
            "POWER_CONCENTRATION", "NO_APPEAL", "VAGUE_ENFORCEMENT",
            "MISSING_QUORUM", "MISSING_TERM_LIMITS", "AMENDMENT_LOCK",
            "CONFLICT_OF_INTEREST", "OPAQUE_PROCESS",
        ]
        cat_filter = st.multiselect("Filter by category", CATEGORIES, default=CATEGORIES)

    filtered = [
        r for r in risks
        if r["severity"] in sev_filter and r["category"] in cat_filter
    ]

    st.subheader(f"Risk Flags ({len(filtered)} shown)")

    SEV_EMOJI = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}

    for risk in filtered:
        emoji = SEV_EMOJI.get(risk["severity"], "⚪")
        with st.expander(
            f"{emoji} [{risk['severity']}] {risk['heading']}  —  {risk['category']}"
        ):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Problem**")
                st.write(risk["reason"])
            with col_b:
                st.markdown("**Suggestion**")
                st.write(risk["suggestion"])
            st.caption(f"Clause ID: `{risk['clause_id']}`  |  Score: {risk['score']:.2f}")


# =============================================================================
# PAGE 4 — New Clause Impact
# =============================================================================
elif page == "🔗 New Clause Impact":
    st.header("🔗 New Clause Impact Analysis")

    if not st.session_state.ingested:
        st.warning("Please ingest a constitution first (page 1).")
        st.stop()

    st.write(
        "Paste a proposed new clause below. "
        "The system will find similar existing clauses, classify relationships, "
        "traverse the dependency graph, and tell you what would need to change."
    )

    col1, col2 = st.columns(2)
    with col1:
        new_heading = st.text_input(
            "Clause heading / title",
            placeholder="e.g. 7.4 Term Limits for Office Bearers",
        )
    with col2:
        commit = st.toggle(
            "Commit to graph after analysis",
            value=False,
            help="If ON, the clause is permanently added to the graph and vector store.",
        )

    new_text = st.text_area(
        "Clause text",
        placeholder="e.g. No office bearer shall serve more than two consecutive terms in the same position.",
        height=150,
    )

    if st.button("🔗 Analyse Impact", type="primary") and new_text.strip():
        if not new_heading.strip():
            new_heading = "Proposed Clause"

        with st.status("Analysing impact...", expanded=True) as status:
            try:
                st.write("🔢 Embedding new clause...")
                st.write("🔍 Searching for similar existing clauses...")
                st.write("🕸️  Classifying relationships via LLM...")
                st.write("🚶 Traversing dependency graph...")

                report = analyse_new_clause(
                    new_clause_text=new_text,
                    new_clause_heading=new_heading,
                    commit_to_graph=commit,
                )

                if "error" in report:
                    status.update(label="❌ Analysis failed", state="error")
                    st.error(report["error"])
                    st.stop()

                status.update(label="✅ Impact analysis complete", state="complete")

            except Exception as e:
                status.update(label="❌ Error", state="error")
                st.error(str(e))
                st.stop()

        # ── LLM plain-English summary ─────────────────────────────────────
        st.subheader("📋 Impact Summary")
        st.info(report.get("llm_summary", "Summary not available."))

        if report.get("committed"):
            st.success("✅ Clause has been committed to the graph and vector store.")

        # ── Metrics row ───────────────────────────────────────────────────
        summary = report.get("summary", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total affected",  summary.get("total_affected", 0))
        c2.metric("Conflicts",       summary.get("conflict_count", 0))
        c3.metric("Dependents",      summary.get("dependent_count", 0))
        c4.metric("Dependencies",    summary.get("dependency_count", 0))

        st.divider()

        # ── Conflicts ─────────────────────────────────────────────────────
        conflicts = report.get("conflicts", [])
        st.subheader(f"🔴 Conflicts ({len(conflicts)})")
        if conflicts:
            for c in conflicts:
                with st.expander(f"{score_emoji(c['score'])} {c['heading']} — {c['label']} conflict"):
                    st.write(c.get("reason", "No reason provided."))
                    st.caption(f"Score: {c['score']:.2f}  |  ID: `{c['id']}`")
                    st.write(c.get("text", "")[:400])
        else:
            st.success("No direct conflicts found.")

        # ── Dependents ────────────────────────────────────────────────────
        dependents = report.get("dependents", [])
        st.subheader(f"⚡ Clauses that would be affected ({len(dependents)})")
        if dependents:
            dep_df = pd.DataFrame([
                {
                    "Heading":  d["heading"],
                    "Section":  d["section"],
                    "Hop":      d["hop"],
                    "Strength": d["label"],
                    "Score":    round(d["score"], 2),
                    "Reason":   d.get("reason", ""),
                }
                for d in dependents
            ])
            st.dataframe(dep_df, use_container_width=True, hide_index=True)
        else:
            st.success("No dependent clauses found.")

        # ── Dependencies (what the new clause needs) ──────────────────────
        dependencies = report.get("dependencies", [])
        st.subheader(f"🔗 What this clause depends on ({len(dependencies)})")
        if dependencies:
            for d in dependencies:
                st.markdown(
                    f"- **{d['heading']}** *(hop {d['hop']}, {d['label']})*  "
                    f"— {d.get('reason', '')}"
                )
        else:
            st.success("This clause has no dependencies on existing clauses.")

        # ── Override relationships ─────────────────────────────────────────
        overrides = report.get("overrides", {})
        if overrides.get("overrides") or overrides.get("overridden_by"):
            st.subheader("📌 Override relationships")
            for ov in overrides.get("overrides", []):
                st.warning(f"This clause **overrides**: {ov['heading']} ({ov['label']})")
            for ov in overrides.get("overridden_by", []):
                st.error(f"This clause is **overridden by**: {ov['heading']} ({ov['label']})")

        # ── Required changes checklist ─────────────────────────────────────
        changes = report.get("change_required", [])
        st.subheader(f"📝 Required changes ({len(changes)})")
        if changes:
            for ch in changes:
                st.checkbox(ch, value=False, key=f"change_{hash(ch)}")
        else:
            st.success("No changes required to existing clauses.")

        # ── Similar clauses ───────────────────────────────────────────────
        with st.expander(f"🔍 Similar existing clauses ({len(report.get('similar', []))})"):
            for s in report.get("similar", []):
                st.markdown(f"**{s['heading']}** — similarity {s['score']:.2f} ({s['label']})")
                st.caption(s.get("text", "")[:300])
                st.divider()


# =============================================================================
# PAGE 5 — Graph Insights
# =============================================================================
elif page == "🕸️ Graph Insights":
    st.header("🕸️ Graph Insights")

    if not st.session_state.ingested:
        st.warning("Please ingest a constitution first (page 1).")
        st.stop()

    G = st.session_state.graph
    if not G:
        st.error("Graph not available. Re-ingest the constitution.")
        st.stop()

    # ── Edge type colour map ──────────────────────────────────────────────
    EDGE_COLORS = {
        "DEPENDS_ON":     "#E05252",   # red
        "CONFLICTS_WITH": "#E08C52",   # orange
        "OVERRIDES":      "#9B52E0",   # purple
        "SIMILAR_TO":     "#52A7E0",   # blue
        "DEFINES":        "#52C075",   # green
        "USES":           "#A0C452",   # yellow-green
    }
    NODE_COLOR_DEFAULT = "#4A90D9"

    # ── Helper: build pyvis HTML ──────────────────────────────────────────
    def _build_pyvis(
        subgraph: nx.DiGraph,
        height: str = "500px",
        risk_map: dict = None,
    ) -> str:
        net = Network(height=height, width="100%", directed=True, bgcolor="#0e1117", font_color="white")
        net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120)

        for node_id, attrs in subgraph.nodes(data=True):
            label   = attrs.get("heading", node_id)[:30]
            title   = f"<b>{attrs.get('heading', node_id)}</b><br>{attrs.get('text','')[:200]}..."
            section = attrs.get("section", "")

            # Size by degree (more connected = bigger)
            degree = subgraph.degree(node_id)
            size   = 12 + min(degree * 3, 30)

            # Colour by risk if risk_map provided
            if risk_map and node_id in risk_map:
                sev = risk_map[node_id]
                color = "#E05252" if sev == "HIGH" else "#E0C052" if sev == "MEDIUM" else "#52C075"
            else:
                color = NODE_COLOR_DEFAULT

            net.add_node(node_id, label=label, title=title, size=size, color=color)

        for u, v, data in subgraph.edges(data=True):
            etype  = data.get("type", "SIMILAR_TO")
            score  = data.get("score", 0.5)
            color  = EDGE_COLORS.get(etype, "#888888")
            title  = f"{etype} ({score:.2f})<br>{data.get('reason','')}"
            width  = 1 + score * 3
            net.add_edge(u, v, color=color, title=title, width=width, label=etype)

        net.set_options("""
        {
          "edges": { "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } },
                     "smooth": { "type": "dynamic" } },
          "interaction": { "hover": true, "tooltipDelay": 100 },
          "physics": { "enabled": true }
        }
        """)
        return net.generate_html()

    # ── Top-level stats ───────────────────────────────────────────────────
    edge_type_counts = {}
    for _, _, d in G.edges(data=True):
        t = d.get("type", "UNKNOWN")
        edge_type_counts[t] = edge_type_counts.get(t, 0) + 1

    cols = st.columns(3 + len(edge_type_counts))
    cols[0].metric("Clauses (nodes)", G.number_of_nodes())
    cols[1].metric("Total edges",     G.number_of_edges())
    cols[2].metric("Connected components",
                   nx.number_weakly_connected_components(G))
    for i, (etype, cnt) in enumerate(sorted(edge_type_counts.items())):
        dot = "🔴" if etype=="CONFLICTS_WITH" else "🟣" if etype=="OVERRIDES" else \
              "🟢" if etype=="DEFINES" else "🟡" if etype=="USES" else \
              "🔵" if etype=="SIMILAR_TO" else "⚪"
        cols[3 + i].metric(f"{dot} {etype}", cnt)

    st.divider()

    # ── Four tabs ─────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Ego Graph",
        "🗺️ Full Graph",
        "🔥 Risk Overlay",
        "📊 Insights & Stats",
    ])

    # ── TAB 1: Ego Graph ──────────────────────────────────────────────────
    with tab1:
        st.subheader("Clause neighbourhood explorer")
        st.caption(
            "Select a clause to see only its direct connections. "
            "Hover nodes and edges for detail. Most useful view for day-to-day analysis."
        )

        clause_ids      = list(G.nodes())
        clause_headings = {n: G.nodes[n].get("heading", n) for n in clause_ids}
        options         = [f"{clause_headings[n]} [{n}]" for n in clause_ids]

        selected_opt = st.selectbox("Select clause", options, key="ego_select")
        selected_id  = clause_ids[options.index(selected_opt)]

        edge_filter = st.multiselect(
            "Show edge types",
            list(EDGE_COLORS.keys()),
            default=list(EDGE_COLORS.keys()),
            key="ego_edge_filter",
        )
        hops = st.slider("Neighbourhood depth (hops)", 1, 3, 1, key="ego_hops")

        # Build ego subgraph
        ego_nodes = {selected_id}
        frontier  = {selected_id}
        for _ in range(hops):
            next_frontier = set()
            for n in frontier:
                for nb in list(G.successors(n)) + list(G.predecessors(n)):
                    if nb not in ego_nodes:
                        ego_nodes.add(nb)
                        next_frontier.add(nb)
            frontier = next_frontier

        ego_sub = G.subgraph(ego_nodes).copy()

        # Filter by edge type
        edges_to_remove = [
            (u, v) for u, v, d in ego_sub.edges(data=True)
            if d.get("type") not in edge_filter
        ]
        ego_sub.remove_edges_from(edges_to_remove)

        if ego_sub.number_of_nodes() == 0:
            st.info("No connections found for this clause with selected filters.")
        else:
            html = _build_pyvis(ego_sub, height="480px")
            components.html(html, height=500, scrolling=False)

            # Detail table below graph
            st.subheader("Edge details")
            rows = []
            for u, v, d in G.out_edges(selected_id, data=True):
                if d.get("type") in edge_filter:
                    rows.append({
                        "Direction":    "→ outgoing",
                        "Other clause": G.nodes[v].get("heading", v),
                        "Type":         d.get("type"),
                        "Score":        round(d.get("score", 0), 2),
                        "Reason":       d.get("reason", d.get("term", "")),
                    })
            for u, v, d in G.in_edges(selected_id, data=True):
                if d.get("type") in edge_filter:
                    rows.append({
                        "Direction":    "← incoming",
                        "Other clause": G.nodes[u].get("heading", u),
                        "Type":         d.get("type"),
                        "Score":        round(d.get("score", 0), 2),
                        "Reason":       d.get("reason", d.get("term", "")),
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── TAB 2: Full Graph ─────────────────────────────────────────────────
    with tab2:
        st.subheader("Full constitution graph")
        st.caption(
            "All clauses and all edges. Use edge type filter to reduce noise. "
            "Drag nodes to rearrange. Scroll to zoom."
        )

        full_edge_filter = st.multiselect(
            "Show edge types",
            list(EDGE_COLORS.keys()),
            default=["DEPENDS_ON", "CONFLICTS_WITH", "OVERRIDES", "DEFINES", "USES"],
            key="full_edge_filter",
        )

        full_sub = G.copy()
        remove   = [(u, v) for u, v, d in full_sub.edges(data=True)
                    if d.get("type") not in full_edge_filter]
        full_sub.remove_edges_from(remove)
        # Remove isolated nodes after edge filter
        isolates = [n for n in full_sub.nodes() if full_sub.degree(n) == 0]
        full_sub.remove_nodes_from(isolates)

        if full_sub.number_of_nodes() == 0:
            st.info("No edges match the selected types.")
        else:
            st.caption(
                f"Showing {full_sub.number_of_nodes()} nodes, "
                f"{full_sub.number_of_edges()} edges "
                f"({len(isolates)} isolated nodes hidden)"
            )
            html = _build_pyvis(full_sub, height="600px")
            components.html(html, height=620, scrolling=False)

    # ── TAB 3: Risk Overlay ───────────────────────────────────────────────
    with tab3:
        st.subheader("Risk heat map on graph")
        st.caption(
            "Nodes coloured by highest risk severity found. "
            "🔴 HIGH  🟡 MEDIUM  🟢 LOW  🔵 No risk flagged. "
            "Run Risk Analysis first to populate this view."
        )

        risks = st.session_state.risk_results
        if not risks:
            st.warning(
                "No risk results cached. Go to ⚠️ Risk Analysis and run it first, "
                "then come back here."
            )
        else:
            # Build clause_id → worst severity map
            sev_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            risk_map  = {}
            for r in risks:
                cid = r["clause_id"]
                sev = r["severity"]
                if cid not in risk_map or sev_order[sev] < sev_order[risk_map[cid]]:
                    risk_map[cid] = sev

            risk_edge_filter = st.multiselect(
                "Show edge types",
                list(EDGE_COLORS.keys()),
                default=["DEPENDS_ON", "CONFLICTS_WITH", "OVERRIDES"],
                key="risk_edge_filter",
            )

            risk_sub = G.copy()
            remove   = [(u, v) for u, v, d in risk_sub.edges(data=True)
                        if d.get("type") not in risk_edge_filter]
            risk_sub.remove_edges_from(remove)

            html = _build_pyvis(risk_sub, height="560px", risk_map=risk_map)
            components.html(html, height=580, scrolling=False)

            # Risk hotspot table
            st.subheader("Risk hotspots")
            hotspots = [
                {
                    "Clause":   G.nodes[cid].get("heading", cid),
                    "Severity": sev,
                    "Degree":   G.degree(cid),
                    "ID":       cid,
                }
                for cid, sev in risk_map.items()
                if G.has_node(cid)
            ]
            hotspots.sort(key=lambda x: (sev_order[x["Severity"]], -x["Degree"]))
            st.dataframe(pd.DataFrame(hotspots), use_container_width=True, hide_index=True)

    # ── TAB 4: Insights & Stats ───────────────────────────────────────────
    with tab4:
        st.subheader("Structural insights")

        col_a, col_b = st.columns(2)

        with col_a:
            # Most connected clauses
            st.markdown("**Most connected clauses** (by total degree)")
            degree_rows = sorted(
                [
                    {
                        "Clause":  G.nodes[n].get("heading", n),
                        "Section": G.nodes[n].get("section", ""),
                        "In":      G.in_degree(n),
                        "Out":     G.out_degree(n),
                        "Total":   G.degree(n),
                    }
                    for n in G.nodes()
                ],
                key=lambda x: -x["Total"],
            )[:15]
            st.dataframe(pd.DataFrame(degree_rows), use_container_width=True, hide_index=True)

        with col_b:
            # Defined terms table
            st.markdown("**Defined terms** (DEFINES edges)")
            def_rows = []
            for u, v, d in G.edges(data=True):
                if d.get("type") == "DEFINES":
                    def_rows.append({
                        "Defined in": G.nodes[u].get("heading", u),
                        "Used in":    G.nodes[v].get("heading", v),
                        "Term":       d.get("term", ""),
                    })
            if def_rows:
                st.dataframe(pd.DataFrame(def_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No DEFINES edges found yet. Re-ingest to detect terminology.")

        st.divider()

        col_c, col_d = st.columns(2)

        with col_c:
            # Conflict pairs
            st.markdown("**Conflict pairs** (CONFLICTS_WITH edges)")
            conflict_rows = [
                {
                    "Clause A": G.nodes[u].get("heading", u),
                    "Clause B": G.nodes[v].get("heading", v),
                    "Score":    round(d.get("score", 0), 2),
                    "Reason":   d.get("reason", ""),
                }
                for u, v, d in G.edges(data=True)
                if d.get("type") == "CONFLICTS_WITH"
            ]
            if conflict_rows:
                st.dataframe(pd.DataFrame(conflict_rows), use_container_width=True, hide_index=True)
            else:
                st.success("No conflicts detected in the constitution.")

        with col_d:
            # Isolated clauses (no edges at all — governance blind spots)
            st.markdown("**Isolated clauses** (no relationships — potential blind spots)")
            isolated = [
                {
                    "Clause":  G.nodes[n].get("heading", n),
                    "Section": G.nodes[n].get("section", ""),
                }
                for n in G.nodes()
                if G.degree(n) == 0
            ]
            if isolated:
                st.dataframe(pd.DataFrame(isolated), use_container_width=True, hide_index=True)
                st.caption(
                    "Isolated clauses have no detected relationships with the rest of the "
                    "constitution. This may indicate standalone rules, or clauses the graph "
                    "failed to connect — worth reviewing manually."
                )
            else:
                st.success("All clauses are connected to at least one other clause.")
