#!/usr/bin/env python3
"""
Live CSI pose detection — Host PC GUI
======================================
Connect to Banana Pi over SSH, stream CSI, run cnn_gru_multitask_sam,
and draw the predicted human skeleton in real time.

Usage:
  python3 live_gui.py
  python3 live_gui.py --checkpoint best_model.pt
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

import numpy as np

from bpi_ssh_worker import BPISshWorker
from csi_preprocessor import LiveCSIBuffer
from pose_engine import DEFAULT_CHECKPOINT, PoseEngine
from skeleton_viz import draw_skeleton_on_canvas, pose_summary

try:
    import paramiko  # noqa: F401
    PARAMIKO_OK = True
except ImportError:
    PARAMIKO_OK = False

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib

    matplotlib.use("TkAgg")
    from skeleton_viz import draw_skeleton_on_axes

    MPL_OK = True
except ImportError:
    MPL_OK = False


C = {
    "bg": "#0d1117",
    "panel": "#161b27",
    "border": "#252d40",
    "text": "#dde1f0",
    "muted": "#5a6080",
    "teal": "#00c9a7",
    "orange": "#ff7043",
    "blue": "#4a9eff",
    "red": "#f44747",
    "green": "#23d18b",
    "yellow": "#ffca28",
    "entry_bg": "#1e2436",
}

FONT_MONO = ("Consolas", 10)
FONT_LABEL = ("Segoe UI", 10)
FONT_HEAD = ("Segoe UI", 11, "bold")
FONT_TINY = ("Segoe UI", 8)

CANVAS_W = 480
CANVAS_H = 640
INFERENCE_HZ = 10


class LivePoseApp(tk.Tk):
    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        self.title("CSI Live Pose Detection")
        self.configure(bg=C["bg"])
        self.minsize(980, 720)
        self.resizable(True, True)

        self.checkpoint = checkpoint
        self.engine: Optional[PoseEngine] = None
        self.csi_buffer = LiveCSIBuffer()

        self.worker: Optional[BPISshWorker] = None
        self.data_q: queue.Queue = queue.Queue()
        self.status_q: queue.Queue = queue.Queue()
        self.connected = False
        self.inference_enabled = False

        self.batch_total = 0
        self.inference_count = 0
        self.last_keypoints: Optional[np.ndarray] = None
        self.last_pose_name: str = ""
        self.last_pose_probs: Optional[np.ndarray] = None
        self.last_infer_t = 0.0
        self._replay_mode = False
        self._pose_votes: List[str] = []
        self.VOTE_WINDOW = 5

        self._build_ui()
        self._load_model()
        self.after(100, self._poll)

    def _load_model(self) -> None:
        try:
            self.engine = PoseEngine(self.checkpoint)
            self.csi_buffer = LiveCSIBuffer(csi_norm=self.engine.csi_norm)
            info = self.engine.info()
            self.model_lbl.configure(
                text=(
                    f"{info.get('model_name', 'multitask')}  "
                    f"epoch={info['epoch']}  device={info['device']}"
                ),
                fg=C["green"],
            )
            self._log("info", f"Loaded checkpoint: {info['checkpoint']}")
            self._log(
                "info",
                f"CSI norm: {info.get('csi_norm_file')}  "
                f"classes={info.get('pose_classes')}",
            )
            if info.get("val_pose_acc") == info.get("val_pose_acc"):
                self._log(
                    "info",
                    f"Checkpoint metrics: val_loss={info['val_loss']:.4f} "
                    f"val_pose_acc={info['val_pose_acc']:.3f}",
                )
        except Exception as exc:
            self.model_lbl.configure(text=f"Model error: {exc}", fg=C["red"])
            self._log("error", str(exc))

    def _section(self, parent: tk.Widget, title: str) -> tk.LabelFrame:
        f = tk.LabelFrame(
            parent,
            text=f"  {title}  ",
            bg=C["panel"],
            fg=C["teal"],
            font=FONT_HEAD,
            bd=0,
            highlightbackground=C["border"],
            highlightthickness=1,
        )
        f.pack(fill=tk.X, padx=6, pady=4)
        return f

    def _btn(self, parent, text, bg, fg, cmd, **kw):
        return tk.Button(
            parent,
            text=text,
            font=FONT_LABEL,
            bg=bg,
            fg=fg,
            activebackground=bg,
            activeforeground=fg,
            relief="flat",
            cursor="hand2",
            command=cmd,
            **kw,
        )

    def _entry_row(self, parent, label, default="", show=None):
        row = tk.Frame(parent, bg=C["panel"])
        row.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(row, text=label, bg=C["panel"], fg=C["muted"], font=FONT_TINY, width=14, anchor="w").pack(side=tk.LEFT)
        var = tk.StringVar(value=default)
        ent = tk.Entry(
            row,
            textvariable=var,
            bg=C["entry_bg"],
            fg=C["text"],
            insertbackground=C["text"],
            relief="flat",
            font=FONT_MONO,
            show=show,
        )
        ent.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)
        return var

    def _build_ui(self) -> None:
        top = tk.Frame(self, bg=C["bg"])
        top.pack(fill=tk.X, padx=8, pady=(8, 2))
        tk.Label(top, text="CSI LIVE POSE DETECTION", bg=C["bg"], fg=C["teal"], font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT)
        self.model_lbl = tk.Label(top, text="Loading model...", bg=C["bg"], fg=C["muted"], font=FONT_TINY)
        self.model_lbl.pack(side=tk.RIGHT)

        pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=C["bg"], sashwidth=5, sashrelief="flat")
        pane.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        left = tk.Frame(pane, bg=C["bg"], width=380)
        pane.add(left, minsize=340)
        right = tk.Frame(pane, bg=C["bg"])
        pane.add(right, minsize=520)

        self._build_conn(left)
        self._build_controls(left)
        self._build_pose_view(right)
        self._build_log(right)
        self._build_statusbar()

    def _build_conn(self, parent) -> None:
        sec = self._section(parent, "Banana Pi Connection")
        self.host_var = self._entry_row(sec, "BPI IP", "192.168.1.219")
        self.user_var = self._entry_row(sec, "SSH user", "root")
        self.pass_var = self._entry_row(sec, "Password", show="*")
        self.port_var = self._entry_row(sec, "SSH port", "22")
        self.iface_var = self._entry_row(sec, "WiFi iface", "wifi0")

        btn_row = tk.Frame(sec, bg=C["panel"])
        btn_row.pack(fill=tk.X, padx=8, pady=8)
        self.connect_btn = self._btn(btn_row, "Connect", C["teal"], "#0d1117", self._toggle_connect, width=12)
        self.connect_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.infer_btn = self._btn(btn_row, "Start Inference", C["blue"], "white", self._toggle_inference, width=14, state=tk.DISABLED)
        self.infer_btn.pack(side=tk.LEFT)

        self.conn_lbl = tk.Label(sec, text="Not connected", bg=C["panel"], fg=C["muted"], font=FONT_TINY, anchor="w")
        self.conn_lbl.pack(fill=tk.X, padx=8, pady=(0, 6))

    def _build_controls(self, parent) -> None:
        sec = self._section(parent, "Replay / Live calibrate")
        tk.Label(
            sec,
            text=(
                "Replay any training CSI JSON (day*/…json).\n"
                "Empty is classified — no live empty residual calibrate."
            ),
            bg=C["panel"],
            fg=C["muted"],
            font=FONT_TINY,
            wraplength=320,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=8, pady=(4, 2))
        self._btn(sec, "Replay CSI JSON", C["orange"], "white", self._replay_json).pack(
            fill=tk.X, padx=8, pady=(6, 4)
        )
        self._btn(
            sec,
            "Calibrate empty room (live)",
            C["green"],
            "#0d1117",
            self._calibrate_empty,
        ).pack(fill=tk.X, padx=8, pady=4)
        self._btn(
            sec,
            "Reset empty residual",
            C["panel"],
            C["text"],
            self._reset_empty,
        ).pack(fill=tk.X, padx=8, pady=(0, 6))
        self.calib_lbl = tk.Label(
            sec,
            text="Empty: classifier (no residual)",
            bg=C["panel"],
            fg=C["green"],
            font=FONT_TINY,
            anchor="w",
            wraplength=320,
            justify=tk.LEFT,
        )
        self.calib_lbl.pack(fill=tk.X, padx=8, pady=(0, 6))

        stats = self._section(parent, "Live Stats")
        self.stats_lbl = tk.Label(
            stats,
            text="Snapshots: 0\nInferences: 0",
            bg=C["panel"],
            fg=C["text"],
            font=FONT_MONO,
            justify=tk.LEFT,
            anchor="w",
        )
        self.stats_lbl.pack(fill=tk.X, padx=8, pady=6)

    def _build_pose_view(self, parent) -> None:
        sec = tk.LabelFrame(
            parent,
            text="  Predicted Skeleton  ",
            bg=C["panel"],
            fg=C["teal"],
            font=FONT_HEAD,
            bd=0,
            highlightbackground=C["border"],
            highlightthickness=1,
        )
        sec.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.pose_summary_lbl = tk.Label(sec, text="Waiting for CSI...", bg=C["panel"], fg=C["muted"], font=FONT_TINY)
        self.pose_summary_lbl.pack(fill=tk.X, padx=8, pady=(6, 2))

        if MPL_OK:
            self.fig = Figure(figsize=(5.2, 6.8), facecolor=C["panel"])
            self.ax = self.fig.add_subplot(111)
            self.ax.set_facecolor(C["bg"])
            self.canvas = FigureCanvasTkAgg(self.fig, master=sec)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            self._draw_empty_plot()
        else:
            self.tk_canvas = tk.Canvas(sec, width=CANVAS_W, height=CANVAS_H, bg=C["bg"], highlightthickness=0)
            self.tk_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            self.tk_canvas.create_text(
                CANVAS_W // 2,
                CANVAS_H // 2,
                text="Install matplotlib for skeleton view",
                fill=C["muted"],
                font=FONT_LABEL,
            )

    def _build_log(self, parent) -> None:
        sec = self._section(parent, "Log")
        self.log_txt = tk.Text(
            sec,
            height=8,
            bg=C["entry_bg"],
            fg=C["text"],
            font=FONT_MONO,
            relief="flat",
            wrap=tk.WORD,
        )
        self.log_txt.pack(fill=tk.BOTH, expand=False, padx=8, pady=6)
        self.log_txt.configure(state=tk.DISABLED)

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self, bg=C["panel"], height=24)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_lbl = tk.Label(bar, text="Ready", bg=C["panel"], fg=C["muted"], font=FONT_TINY, anchor="w")
        self.status_lbl.pack(fill=tk.X, padx=8, pady=4)

    def _draw_empty_plot(self) -> None:
        if not MPL_OK:
            return
        self.ax.clear()
        self.ax.set_facecolor(C["bg"])
        self.ax.text(0.5, 0.5, "Waiting for CSI...", color=C["muted"], ha="center", va="center", transform=self.ax.transAxes)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(1, 0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.canvas.draw_idle()

    def _render_pose(self, keypoints: np.ndarray, pose_name: str = "", pose_probs=None) -> None:
        title = f"Predicted: {pose_name}" if pose_name else "Predicted pose"
        if pose_name in ("empty", "empty_room") or np.allclose(keypoints, 0, atol=0.01):
            if MPL_OK:
                self.ax.clear()
                self.ax.set_facecolor(C["bg"])
                self.ax.text(
                    0.5, 0.5, "empty room",
                    color=C["muted"], ha="center", va="center",
                    transform=self.ax.transAxes, fontsize=14,
                )
                self.ax.set_xlim(0, 1)
                self.ax.set_ylim(1, 0)
                self.ax.set_xticks([])
                self.ax.set_yticks([])
                for spine in self.ax.spines.values():
                    spine.set_visible(False)
                self.canvas.draw_idle()
            else:
                self.tk_canvas.delete("all")
                self.tk_canvas.create_text(
                    CANVAS_W // 2, CANVAS_H // 2,
                    text="empty room", fill=C["muted"], font=FONT_LABEL,
                )
        elif MPL_OK:
            self.ax.clear()
            draw_skeleton_on_axes(self.ax, keypoints, title=title)
            self.canvas.draw_idle()
        else:
            draw_skeleton_on_canvas(self.tk_canvas, keypoints, CANVAS_W, CANVAS_H)
        self.pose_summary_lbl.configure(
            text=pose_summary(keypoints, pose_name or None, pose_probs),
            fg=C["text"],
        )

    def _log(self, level: str, msg: str) -> None:
        colors = {"info": C["text"], "ok": C["green"], "warn": C["yellow"], "error": C["red"]}
        ts = time.strftime("%H:%M:%S")
        self.log_txt.configure(state=tk.NORMAL)
        self.log_txt.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_txt.see(tk.END)
        self.log_txt.configure(state=tk.DISABLED)
        self.status_lbl.configure(text=msg, fg=colors.get(level, C["text"]))

    def _toggle_connect(self) -> None:
        if self.connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self) -> None:
        if not PARAMIKO_OK:
            messagebox.showerror("Missing dependency", "Install paramiko: pip install paramiko")
            return

        host = self.host_var.get().strip()
        user = self.user_var.get().strip()
        if not host or not user:
            messagebox.showwarning("Input required", "Enter BPI IP and SSH username.")
            return

        try:
            port = int(self.port_var.get().strip() or "22")
        except ValueError:
            messagebox.showwarning("Invalid port", "SSH port must be a number.")
            return

        self.csi_buffer = LiveCSIBuffer(
            csi_norm=self.engine.csi_norm if self.engine else None
        )
        self.worker = BPISshWorker(
            host=host,
            username=user,
            password=self.pass_var.get(),
            wifi_if=self.iface_var.get().strip() or "wifi0",
            ssh_port=port,
            data_queue=self.data_q,
            status_queue=self.status_q,
        )
        self.worker.start()
        self.connected = True
        self.connect_btn.configure(text="Disconnect", bg=C["red"], fg="white")
        self.infer_btn.configure(state=tk.NORMAL)
        self.conn_lbl.configure(text=f"Connecting to {host}...", fg=C["yellow"])
        self._log("info", f"Connecting to {host}:{port}...")

    def _disconnect(self) -> None:
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.connected = False
        self.inference_enabled = False
        self.connect_btn.configure(text="Connect", bg=C["teal"], fg="#0d1117")
        self.infer_btn.configure(text="Start Inference", bg=C["blue"])
        self.infer_btn.configure(state=tk.DISABLED)
        self.conn_lbl.configure(text="Disconnected", fg=C["muted"])
        self._log("info", "Disconnected")

    def _calibrate_empty(self) -> None:
        messagebox.showinfo(
            "Calibration not needed",
            "This multitask model uses amplitude + train-set z-score.\n"
            "'empty' is predicted by the classifier — no live empty residual.",
        )

    def _reset_empty(self) -> None:
        if not self.engine:
            return
        self.engine.reset_empty_mean()
        self.calib_lbl.configure(
            text="Empty: classifier (no residual)",
            fg=C["green"],
        )
        self._log("info", "Empty residual N/A for this model")

    def _vote_pose(self, pose_name: str) -> str:
        self._pose_votes.append(pose_name)
        if len(self._pose_votes) > self.VOTE_WINDOW:
            self._pose_votes = self._pose_votes[-self.VOTE_WINDOW :]
        # Majority vote; prefer most recent on ties.
        counts: dict[str, int] = {}
        for p in self._pose_votes:
            counts[p] = counts.get(p, 0) + 1
        best = max(counts.values())
        for p in reversed(self._pose_votes):
            if counts[p] == best:
                return p
        return pose_name

    def _toggle_inference(self) -> None:
        if not self.engine:
            messagebox.showerror("Model not loaded", "Cannot run inference without a loaded model.")
            return
        self.inference_enabled = not self.inference_enabled
        if self.inference_enabled:
            self._replay_mode = False
            self.csi_buffer = LiveCSIBuffer(csi_norm=self.engine.csi_norm)
            self._pose_votes = []
            self.infer_btn.configure(text="Stop Inference", bg=C["red"])
            self._log("ok", "Live inference started (buffer cleared, burst windows)")
        else:
            self.infer_btn.configure(text="Start Inference", bg=C["blue"])
            self._log("info", "Live inference stopped")

    def _replay_json(self) -> None:
        if not self.engine:
            messagebox.showerror("Model not loaded", "Load the model first.")
            return

        path = filedialog.askopenfilename(
            title="Select CSI JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(Path(__file__).resolve().parent.parent / "data" / "raw" / "csi"),
        )
        if not path:
            return

        try:
            # Freeze live stream overwriting the display while we evaluate the file.
            was_inferring = self.inference_enabled
            self.inference_enabled = False
            self._replay_mode = True

            with open(path, "r") as f:
                data = json.load(f)
            records = data.get("records", [])
            if not records:
                raise ValueError("No records found in JSON")

            self.csi_buffer = LiveCSIBuffer(csi_norm=self.engine.csi_norm)
            added = self.csi_buffer.ingest_json_records(records)
            windows = self.csi_buffer.iter_burst_inputs()
            self._log(
                "info",
                f"Replay {Path(path).name}: {len(records)} records → "
                f"{added} snapshots → {len(windows)} burst windows",
            )

            if not windows:
                raise ValueError(
                    "No full CSI windows yet (need 5 bursts × ~5 RX snapshots). "
                    f"records_ingested≈{len(self.csi_buffer.snapshots)}"
                )

            from collections import Counter

            votes = Counter()
            last_pred = None
            for sample in windows:
                last_pred = self.engine.predict(sample)
                votes[last_pred["pose_name"]] += 1

            majority = votes.most_common(1)[0][0]
            # Prefer last window that matches majority for skeleton display.
            display = last_pred
            for sample in reversed(windows):
                pred = self.engine.predict(sample)
                if pred["pose_name"] == majority:
                    display = pred
                    break

            self.last_keypoints = display["keypoints"]
            self.last_pose_name = majority
            self.last_pose_probs = display["pose_probs"]
            self._render_pose(display["keypoints"], majority, display["pose_probs"])
            self.inference_count += len(windows)
            self._update_stats()
            vote_txt = ", ".join(f"{k}={v}" for k, v in votes.most_common())
            self._log("ok", f"Replay majority={majority}  ({vote_txt})")

            if was_inferring:
                self.infer_btn.configure(text="Start Inference", bg=C["blue"])
                self._log("info", "Live inference was paused for replay — press Start Inference again")
        except Exception as exc:
            messagebox.showerror("Replay failed", str(exc))
            self._log("error", f"Replay failed: {exc}")
        finally:
            self._replay_mode = False

    def _maybe_infer(self) -> None:
        if (
            not self.inference_enabled
            or self._replay_mode
            or not self.engine
            or not self.csi_buffer.ready
        ):
            return

        now = time.time()
        if now - self.last_infer_t < 1.0 / INFERENCE_HZ:
            return

        sample = self.csi_buffer.latest_input()
        if sample is None:
            return

        try:
            pred = self.engine.predict(sample)
            voted = self._vote_pose(pred["pose_name"])
            # If vote differs, still show voted class skeleton from this frame
            # only when classes match; otherwise keep keypoints but label vote.
            self.last_keypoints = pred["keypoints"]
            self.last_pose_name = voted
            self.last_pose_probs = pred["pose_probs"]
            self.inference_count += 1
            self.last_infer_t = now
            self._render_pose(pred["keypoints"], voted, pred["pose_probs"])
            self._update_stats()
            if self.inference_count % 20 == 1:
                stats = self.csi_buffer.amp_stats() or {}
                conf = float(np.max(pred["pose_probs"]))
                self._log(
                    "info",
                    f"live raw_pose={pred['pose_name']} voted={voted} "
                    f"conf={conf:.0%} bursts={stats.get('bursts_ready')} "
                    f"snaps={stats.get('snapshots')}",
                )
        except Exception as exc:
            self._log("error", f"Inference error: {exc}")

    def _update_stats(self) -> None:
        pose = self.last_pose_name or "—"
        conf = ""
        if self.last_pose_probs is not None:
            conf = f" ({float(np.max(self.last_pose_probs)):.0%})"
        stats = self.csi_buffer.amp_stats()
        extra = ""
        if stats:
            extra = (
                f"\nBursts ready: {stats['bursts_ready']}\n"
                f"Input μ={stats['norm_mean']:.3f}  "
                f"shape={stats.get('shape')}"
            )
        self.stats_lbl.configure(
            text=(
                f"Snapshots buffered: {len(self.csi_buffer.snapshots)}\n"
                f"Buffer ready: {self.csi_buffer.ready}\n"
                f"Last pose: {pose}{conf}\n"
                f"Inferences: {self.inference_count}\n"
                f"CSI batches: {self.batch_total}"
                f"{extra}"
            )
        )

    def _poll(self) -> None:
        while True:
            try:
                level, msg = self.status_q.get_nowait()
                self._log(level if level in ("info", "ok", "warn", "error") else "info", msg)
                if level == "ok" and self.worker and self.worker.connected:
                    self.conn_lbl.configure(text=f"Connected — {self.host_var.get()}", fg=C["green"])
            except queue.Empty:
                break

        while True:
            try:
                batch = self.data_q.get_nowait()
                if self._replay_mode:
                    continue
                self.batch_total += 1
                self.csi_buffer.ingest_worker_batch(batch)
                self._update_stats()
            except queue.Empty:
                break

        self._maybe_infer()
        self.after(80, self._poll)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live CSI pose detection GUI")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    app = LivePoseApp(args.checkpoint)
    app.mainloop()


if __name__ == "__main__":
    main()
