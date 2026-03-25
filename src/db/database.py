"""
SQLite database layer for MiniSim persistence.

Tables:
- predictions: every swarm prediction with timestamp, scores, resolution
- agent_performance: per-archetype accuracy tracking over time
- market_snapshots: cached market data from platforms
- scan_history: scanner run logs

Usage:
    from src.database import Database
    db = Database()
    db.log_prediction(question, swarm_p, market_p, source, ...)
    db.resolve(pred_id, 1.0)
    metrics = db.get_metrics()
"""
from __future__ import annotations

import json
import os
import sqlite3
import statistics
from datetime import datetime


DB_PATH = "results/minisim.db"


class Database:
    def __init__(self, path: str = DB_PATH):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                question TEXT NOT NULL,
                swarm_probability REAL NOT NULL,
                market_price REAL,
                edge REAL,
                source TEXT,
                ticker TEXT,
                category TEXT,
                n_agents INTEGER,
                n_rounds INTEGER,
                mode TEXT,
                confidence_low REAL,
                confidence_high REAL,
                diversity_score REAL,
                resolution REAL,
                resolved_at TEXT,
                swarm_brier REAL,
                market_brier REAL,
                swarm_beat_market INTEGER
            );

            CREATE TABLE IF NOT EXISTS agent_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                archetype TEXT NOT NULL,
                category TEXT,
                prediction_id INTEGER,
                initial_score REAL,
                final_score REAL,
                actual_resolution REAL,
                brier_score REAL,
                timestamp TEXT,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            );

            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                ticker TEXT,
                question TEXT,
                price REAL,
                volume REAL,
                category TEXT,
                status TEXT
            );

            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                sources TEXT,
                markets_scanned INTEGER,
                opportunities_found INTEGER,
                best_edge REAL,
                config TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_predictions_source ON predictions(source);
            CREATE INDEX IF NOT EXISTS idx_predictions_resolved ON predictions(resolution);
            CREATE INDEX IF NOT EXISTS idx_agent_perf_archetype ON agent_performance(archetype);
            CREATE INDEX IF NOT EXISTS idx_market_source ON market_snapshots(source);
        """)
        self.conn.commit()

    # ── Predictions ──

    def log_prediction(
        self,
        question: str,
        swarm_probability: float,
        market_price: float | None = None,
        source: str = "",
        ticker: str = "",
        category: str = "",
        n_agents: int = 0,
        n_rounds: int = 0,
        mode: str = "offline",
        confidence_interval: list | None = None,
        diversity_score: float = 0,
        agents: list[dict] | None = None,
    ) -> int:
        """Log a prediction. Returns the prediction ID."""
        ci = confidence_interval or []
        edge = swarm_probability - market_price if market_price is not None else None

        cursor = self.conn.execute("""
            INSERT INTO predictions (
                timestamp, question, swarm_probability, market_price, edge,
                source, ticker, category, n_agents, n_rounds, mode,
                confidence_low, confidence_high, diversity_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat() + "Z",
            question, swarm_probability, market_price, edge,
            source, ticker, category, n_agents, n_rounds, mode,
            ci[0] if len(ci) > 0 else None,
            ci[1] if len(ci) > 1 else None,
            diversity_score,
        ))
        pred_id = cursor.lastrowid
        self.conn.commit()

        # Log agent performance
        if agents:
            for a in agents:
                self.conn.execute("""
                    INSERT INTO agent_performance (
                        archetype, category, prediction_id, initial_score,
                        final_score, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    a.get("background_category", ""),
                    category,
                    pred_id,
                    a.get("initial_score"),
                    a.get("final_score"),
                    datetime.utcnow().isoformat() + "Z",
                ))
            self.conn.commit()

        return pred_id

    def resolve(self, pred_id: int, resolution: float) -> bool:
        """Resolve a prediction. Computes Brier scores."""
        row = self.conn.execute(
            "SELECT swarm_probability, market_price FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()

        if not row:
            return False

        swarm_brier = (row["swarm_probability"] - resolution) ** 2
        market_brier = (row["market_price"] - resolution) ** 2 if row["market_price"] is not None else None
        beat = (swarm_brier < market_brier) if market_brier is not None else None

        self.conn.execute("""
            UPDATE predictions SET
                resolution = ?, resolved_at = ?, swarm_brier = ?,
                market_brier = ?, swarm_beat_market = ?
            WHERE id = ?
        """, (
            resolution, datetime.utcnow().isoformat() + "Z",
            round(swarm_brier, 6), round(market_brier, 6) if market_brier else None,
            1 if beat else (0 if beat is not None else None),
            pred_id,
        ))

        # Update agent performance
        self.conn.execute("""
            UPDATE agent_performance SET
                actual_resolution = ?,
                brier_score = (final_score - ?) * (final_score - ?)
            WHERE prediction_id = ?
        """, (resolution, resolution, resolution, pred_id))

        self.conn.commit()
        return True

    def get_predictions(self, resolved_only: bool = False, limit: int = 100) -> list[dict]:
        """Get predictions as list of dicts."""
        where = "WHERE resolution IS NOT NULL" if resolved_only else ""
        rows = self.conn.execute(
            f"SELECT * FROM predictions {where} ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_metrics(self) -> dict:
        """Compute overall accuracy metrics."""
        total = self.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        resolved = self.conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE resolution IS NOT NULL"
        ).fetchone()[0]

        if resolved == 0:
            return {"total": total, "resolved": 0, "pending": total}

        rows = self.conn.execute(
            "SELECT swarm_brier, market_brier, swarm_beat_market FROM predictions WHERE resolution IS NOT NULL"
        ).fetchall()

        swarm_briers = [r["swarm_brier"] for r in rows if r["swarm_brier"] is not None]
        market_briers = [r["market_brier"] for r in rows if r["market_brier"] is not None]
        wins = sum(1 for r in rows if r["swarm_beat_market"] == 1)

        return {
            "total": total,
            "resolved": resolved,
            "pending": total - resolved,
            "swarm_brier": round(statistics.mean(swarm_briers), 4) if swarm_briers else None,
            "market_brier": round(statistics.mean(market_briers), 4) if market_briers else None,
            "win_rate": round(wins / resolved, 4) if resolved > 0 else None,
            "wins": wins,
        }

    def get_archetype_accuracy(self) -> list[dict]:
        """Get accuracy metrics per agent archetype."""
        rows = self.conn.execute("""
            SELECT archetype,
                   COUNT(*) as n_predictions,
                   AVG(brier_score) as avg_brier,
                   MIN(brier_score) as best_brier,
                   MAX(brier_score) as worst_brier
            FROM agent_performance
            WHERE brier_score IS NOT NULL
            GROUP BY archetype
            ORDER BY avg_brier ASC
        """).fetchall()
        return [dict(r) for r in rows]

    def get_category_accuracy(self) -> list[dict]:
        """Get accuracy metrics per question category."""
        rows = self.conn.execute("""
            SELECT category,
                   COUNT(*) as n,
                   AVG(swarm_brier) as avg_swarm_brier,
                   AVG(market_brier) as avg_market_brier,
                   SUM(swarm_beat_market) as wins
            FROM predictions
            WHERE resolution IS NOT NULL AND category != ''
            GROUP BY category
            ORDER BY avg_swarm_brier ASC
        """).fetchall()
        return [dict(r) for r in rows]

    # ── Market snapshots ──

    def snapshot_market(self, source: str, ticker: str, question: str,
                       price: float, volume: float = 0, category: str = "", status: str = ""):
        self.conn.execute("""
            INSERT INTO market_snapshots (timestamp, source, ticker, question, price, volume, category, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (datetime.utcnow().isoformat() + "Z", source, ticker, question, price, volume, category, status))
        self.conn.commit()

    # ── Scan history ──

    def log_scan(self, sources: list[str], markets_scanned: int,
                 opportunities_found: int, best_edge: float, config: dict):
        self.conn.execute("""
            INSERT INTO scan_history (timestamp, sources, markets_scanned, opportunities_found, best_edge, config)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat() + "Z",
            ",".join(sources), markets_scanned, opportunities_found,
            best_edge, json.dumps(config),
        ))
        self.conn.commit()

    def close(self):
        self.conn.close()
