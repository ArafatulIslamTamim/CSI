#!/usr/bin/env python3
"""
SSH CSI worker — polls mt76-vendor on Banana Pi from the host PC.

No stream server or extra scripts on BPI; only SSH + mt76-vendor/iw.
"""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from typing import List, Optional

import paramiko

VENDOR_CMD = "/usr/sbin/mt76-vendor"
STA_FILE_TMPL = "/tmp/bpi-csi-state/{}.sta"
ARM_SETTLE_S = 0.5
REARM_PERIOD_S = 2.0


def parse_csi_json(raw_text: str) -> list:
    """Parse mt76-vendor dump output (single or concatenated JSON arrays)."""
    raw = raw_text.strip()
    if not raw or raw == "[]":
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed:
            if isinstance(parsed[0], list):
                return parsed
            return [parsed]
    except json.JSONDecodeError:
        pass

    frames: list = []
    depth = 0
    start = 0
    for i, char in enumerate(raw):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                chunk = raw[start : i + 1]
                start = i + 1
                try:
                    parsed = json.loads(chunk)
                    if not parsed:
                        continue
                    if isinstance(parsed[0], list):
                        frames.extend(parsed)
                    else:
                        frames.append(parsed)
                except json.JSONDecodeError:
                    pass
    if frames:
        return frames
    match = re.search(r"(\[.*\])", raw, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return []


def list_record_to_dict(rec, host_ts: Optional[float] = None) -> Optional[dict]:
    """Convert mt76 list record to a structured dict (I/Q unchanged)."""
    if isinstance(rec, dict):
        out = dict(rec)
        out.setdefault("_host_ts", host_ts or time.time())
        return out
    if not isinstance(rec, list) or len(rec) < 13:
        return None
    I, Q = rec[11], rec[12]
    if not isinstance(I, list) or not isinstance(Q, list) or not I or not Q:
        return None
    rx = 0
    try:
        rx = int(rec[8])
    except (TypeError, ValueError, IndexError):
        pass
    return {
        "I": I,
        "Q": Q,
        "rx_ant": rx,
        "n_sub": len(I),
        "_host_ts": host_ts or time.time(),
        "_mac": str(rec[1]) if len(rec) > 1 else "",
        "_raw": rec,
    }


class BPISshWorker(threading.Thread):
    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        wifi_if: str = "wifi0",
        ssh_port: int = 22,
        pkt_num: int = 120,
        interval_ms: int = 100,
        csi_hz: int = 20,
        data_queue: Optional[queue.Queue] = None,
        status_queue: Optional[queue.Queue] = None,
    ):
        super().__init__(daemon=True, name="BPISshWorker")
        self.host = host
        self.username = username
        self.password = password
        self.wifi_if = wifi_if.strip()
        self.ssh_port = ssh_port
        self.pkt_num = max(1, int(pkt_num))
        self.interval_ms = max(10, int(interval_ms))
        self.csi_hz = max(1, int(csi_hz))
        self.data_queue = data_queue or queue.Queue(maxsize=50000)
        self.status_queue = status_queue or queue.Queue(maxsize=500)
        self.discovered_stas: List[str] = []
        self._stop_evt = threading.Event()
        self._ssh: Optional[paramiko.SSHClient] = None
        self.connected = False
        self.csi_armed = False
        self.actual_rps = 0.0

    def stop(self) -> None:
        self._stop_evt.set()

    def _put_status(self, level: str, msg: str) -> None:
        try:
            self.status_queue.put_nowait((level, msg))
        except queue.Full:
            pass

    def _exec(self, cmd: str, timeout: float = 30.0) -> tuple[int, str, str]:
        if not self._ssh:
            raise RuntimeError("SSH not connected")
        _, stdout, stderr = self._ssh.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode(errors="replace").strip()
        err = stderr.read().decode(errors="replace").strip()
        code = stdout.channel.recv_exit_status()
        return code, out, err

    def _connect_ssh(self) -> None:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connected = False
        if self.password:
            try:
                ssh.connect(
                    self.host,
                    port=self.ssh_port,
                    username=self.username,
                    password=self.password,
                    timeout=15,
                    look_for_keys=False,
                    allow_agent=False,
                )
                connected = True
            except Exception:
                pass

        if not connected:
            agent = paramiko.Agent()
            for akey in agent.get_keys():
                try:
                    ssh.connect(
                        self.host,
                        port=self.ssh_port,
                        username=self.username,
                        pkey=akey,
                        timeout=15,
                        look_for_keys=False,
                        allow_agent=False,
                    )
                    connected = True
                    break
                except Exception:
                    pass

        if not connected:
            key_files = [
                os.path.expanduser("~/.ssh/id_ed25519"),
                os.path.expanduser("~/.ssh/id_rsa"),
                os.path.expanduser("~/.ssh/id_ecdsa"),
            ]
            for kf in key_files:
                if not os.path.exists(kf):
                    continue
                try:
                    ssh.connect(
                        self.host,
                        port=self.ssh_port,
                        username=self.username,
                        key_filename=kf,
                        timeout=15,
                        look_for_keys=False,
                        allow_agent=False,
                    )
                    connected = True
                    break
                except Exception:
                    pass

        if not connected:
            raise RuntimeError(
                "SSH authentication failed — check host, port, user, and password/key"
            )

        self._ssh = ssh
        self.connected = True
        self._put_status("info", f"SSH connected to {self.host}:{self.ssh_port}")

    def _station_dump(self, iface: str) -> List[str]:
        stas: List[str] = []
        sta_file = STA_FILE_TMPL.format(iface)
        _, out, _ = self._exec(f"cat {sta_file} 2>/dev/null")
        mac = out.strip()
        if mac and re.match(r"^([0-9a-f]{2}:){5}[0-9a-f]{2}$", mac, re.I):
            stas.append(mac.lower())
        _, out, _ = self._exec(f"iw dev {iface} station dump")
        for m in re.findall(r"^Station\s+([0-9a-f:]{17})", out, re.MULTILINE | re.IGNORECASE):
            if m.lower() not in stas:
                stas.append(m.lower())
        return stas

    def _arm_iface(self, iface: str, stas: List[str]) -> None:
        cmds = [
            f"{VENDOR_CMD} {iface} set csi ctrl=0,0,0,0",
            f"{VENDOR_CMD} {iface} set csi ctrl=1,0,0,0",
        ]
        for sta in stas:
            cmds.append(f"{VENDOR_CMD} {iface} set csi ctrl=2,8,0,1 mac_addr={sta}")
            cmds.append(f"{VENDOR_CMD} {iface} set csi ctrl=2,9,0,1 mac_addr={sta}")
        cmds.append(f"{VENDOR_CMD} {iface} set csi ctrl=2,9,0,{self.csi_hz}")
        cmds.extend([
            f"{VENDOR_CMD} {iface} set csi ctrl=2,3,0,255",
            f"{VENDOR_CMD} {iface} set csi ctrl=2,3,1,255",
        ])
        script = " && sleep 0.15 && ".join(cmds)
        code, out, err = self._exec(script)
        if code != 0 or err:
            self._put_status("warn", f"{iface} CSI arm exit={code}" + (f": {err}" if err else ""))
        if out:
            self._put_status("info", f"{iface} arm: {out[:100]}")

        verify = (
            f"rm -f /tmp/csi_arm_check.json && "
            f"{VENDOR_CMD} {iface} dump csi 10 /tmp/csi_arm_check.json && "
            f"head -c 200 /tmp/csi_arm_check.json"
        )
        _, sample, _ = self._exec(verify)
        if sample and sample != "[]":
            self._put_status("ok", f"{iface} CSI verify OK")
        else:
            self._put_status(
                "warn",
                f"{iface} verify dump empty — ensure STA traffic (e.g. ping AP)",
            )
        time.sleep(ARM_SETTLE_S)
        self.csi_armed = True

    def _stop_csi(self) -> None:
        if not self._ssh:
            return
        try:
            self._exec(f"{VENDOR_CMD} {self.wifi_if} set csi ctrl=0,0,0,0", timeout=10)
        except Exception:
            pass

    def run(self) -> None:
        try:
            self._connect_ssh()
        except Exception as exc:
            self._put_status("error", f"SSH connect failed: {exc}")
            return

        self._put_status("info", f"Scanning STAs on {self.wifi_if}...")
        stas = self._station_dump(self.wifi_if)
        self.discovered_stas = stas
        if not stas:
            self._put_status(
                "error",
                f"No associated STAs on {self.wifi_if} — connect a client first",
            )
            self._shutdown()
            return

        self._put_status("info", f"Found {len(stas)} STA(s): {', '.join(stas)}")
        try:
            self._arm_iface(self.wifi_if, stas)
        except Exception as exc:
            self._put_status("error", f"CSI arm failed: {exc}")
            self._shutdown()
            return

        hz = 1000 // self.csi_hz if self.csi_hz else 0
        self._put_status(
            "ok",
            f"CSI armed @ ~{hz} Hz on {self.wifi_if} for {', '.join(stas)}",
        )

        interval_s = self.interval_ms / 1000.0
        empty_streak = 0
        rate_t0 = time.time()
        rate_n = 0
        last_rearm_t = 0.0
        sta_map = list(stas)

        while not self._stop_evt.is_set():
            loop_start = time.time()

            if loop_start - last_rearm_t >= REARM_PERIOD_S:
                live = self._station_dump(self.wifi_if)
                if not live:
                    self._put_status("warn", f"No STAs on {self.wifi_if} — waiting...")
                elif live != sta_map:
                    sta_map = live
                    self.discovered_stas = live
                    self._arm_iface(self.wifi_if, live)
                    self._put_status("ok", f"Re-armed for: {', '.join(live)}")
                elif empty_streak >= int(REARM_PERIOD_S / max(interval_s, 0.05)):
                    self._arm_iface(self.wifi_if, sta_map)
                    self._put_status("info", "Re-armed CSI (no records recently)")
                last_rearm_t = loop_start

            tmp = f"/tmp/csi_paper5_{self.wifi_if}.json"
            dump_cmd = (
                f"rm -f {tmp} && "
                f"{VENDOR_CMD} {self.wifi_if} dump csi {self.pkt_num} {tmp} && "
                f"test -s {tmp} && cat {tmp}"
            )
            try:
                code, raw, err = self._exec(dump_cmd, timeout=15)
                records = parse_csi_json(raw) if raw else []
                if records:
                    empty_streak = 0
                    # Match csi_video_collector: unique host_ts per record so
                    # RX0/1/2 groups and multi-snapshot bursts survive.
                    parsed = []
                    for rec in records:
                        host_ts = time.time_ns() / 1e9
                        d = list_record_to_dict(rec, host_ts)
                        if d:
                            parsed.append(d)
                    if parsed:
                        try:
                            self.data_queue.put_nowait(parsed)
                        except queue.Full:
                            try:
                                self.data_queue.get_nowait()
                            except queue.Empty:
                                pass
                            try:
                                self.data_queue.put_nowait(parsed)
                            except queue.Full:
                                pass
                    rate_n += len(records)
                    now = time.time()
                    elapsed = now - rate_t0
                    if elapsed >= 1.0:
                        self.actual_rps = rate_n / elapsed
                        self._put_status(
                            "ok",
                            f"{len(records)} records · ~{self.actual_rps:.0f} rec/s",
                        )
                        rate_t0 = now
                        rate_n = 0
                else:
                    empty_streak += 1
                    if code != 0 or err.strip():
                        self._put_status("warn", f"Empty dump: {err.strip() or raw[:80]}")
            except Exception as exc:
                self._put_status("error", f"SSH dump error: {exc}")
                break

            if self._stop_evt.is_set():
                break
            spent = time.time() - loop_start
            time.sleep(max(0.0, interval_s - spent))

        self._shutdown()

    def _shutdown(self) -> None:
        self._stop_csi()
        if self._ssh:
            try:
                self._ssh.close()
            except Exception:
                pass
            self._ssh = None
        self.connected = False
        self.csi_armed = False
        self._put_status("info", "Disconnected")
